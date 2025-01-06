use core::num;
use std::{cmp::min, fs::File, io::{self, BufReader, BufWriter, Write}, path::PathBuf, sync::{atomic::{AtomicBool, Ordering}, mpsc::{self, SyncSender}, Arc, Mutex}, thread::{self, sleep, JoinHandle}, time::{self, Duration, Instant}};

use console::Term;
use cpal::{traits::{HostTrait, StreamTrait}, BufferSize, FromSample, Sample, Stream, StreamConfig};
use hound::{WavSpec, WavWriter};
use rodio::{buffer::SamplesBuffer, DeviceTrait, OutputStream, OutputStreamHandle, Source};
use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType};
use swap_buffer_queue::{buffer::{BufferSlice, VecBuffer}, error::TryDequeueError, Queue};
use termion::{cursor::DetectCursorPos, raw::IntoRawMode};
use whisper_rs::{DtwModelPreset, DtwParameters, FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState};
use once_cell::sync::Lazy;
use crossbeam::channel::{unbounded, Receiver, Sender};

use core_media_rs::cm_sample_buffer::CMSampleBuffer;
use screencapturekit::{
    shareable_content::SCShareableContent,
    stream::{
        configuration::SCStreamConfiguration, content_filter::SCContentFilter,
        output_trait::SCStreamOutputTrait, output_type::SCStreamOutputType, SCStream,
    },
};
use clap::{Command, Arg};
use std::process;
use voice_activity_detector::{Error, IteratorExt, LabeledAudio, VoiceActivityDetector};

//TODO: https://pytorch.org/audio/master/tutorials/forced_alignment_tutorial.html


// use std::{
//     fs::OpenOptions,
//     io::Write,
//     sync::mpsc::{channel, Sender},
//     thread,
//     time::Duration,
// };

// struct ErrorHandler;
// impl StreamErrorHandler for ErrorHandler {
//     fn on_error(&self) {
//         println!("Error!");
//     }
// }

// Define the structure for the audio file writer
pub struct AudioFileWriter {
    writer: Arc<Mutex<WavWriter<BufWriter<File>>>>,
}

impl AudioFileWriter {
    pub fn new(file_path: &str, sample_rate: u32, channels: u16) -> Self {
        let spec = WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let writer = WavWriter::create(file_path, spec).unwrap();
        AudioFileWriter {
            writer: Arc::new(Mutex::new(writer)),
        }
    }

    pub fn write_samples(&self, samples: &[f32]) {
        let mut writer = self.writer.lock().unwrap();
        for &sample in samples {

            writer.write_sample(f32_to_i16(sample)).unwrap();
        }
    }
}

pub struct CapturerWrapper {
    capturer: Arc<Mutex<Capturer>>
}

impl SCStreamOutputTrait for CapturerWrapper {
    fn did_output_sample_buffer(&self, sample_buffer: CMSampleBuffer, of_type: SCStreamOutputType) {
        let start_time = Instant::now();
        self.capturer.lock().unwrap().did_output_sample_buffer(sample_buffer, of_type);
        let duration = start_time.elapsed();
        // println!("Execution time: {:?}", duration);
    }
}

trait AudioSource {
    fn start(&self) -> Result<(), anyhow::Error>;
    fn stop(&mut self) -> Result<(), anyhow::Error>;
    fn get_queue_len(&self) -> usize;
    fn drain_queue(&self) -> Result<Vec<f32>, anyhow::Error>;
}

struct AudioAsyncMic {
    queue: Arc<Queue<VecBuffer<f32>>>,
    processing_handle: Option<JoinHandle<()>>,
    stream: Option<Stream>,
    tx: Option<SyncSender<Vec<f32>>>,
}

impl AudioAsyncMic {
    fn new() -> Result<Self, anyhow::Error> {
        // Use the default audio input device.
        let host = cpal::default_host();
        let device = host.default_input_device().expect("No input device available");
        println!("Using input device: {}", device.name()?);

        // Query supported configurations.
        let supported_configs = device
            .supported_input_configs()
            .expect("Failed to get supported input configurations");

        // Choose a compatible configuration.
        let supported_config = supported_configs
            .filter(|config| config.channels() == 1) // Prefer mono if available
            .next()
            .expect("No compatible configuration found");

        let actual_sample_rate = supported_config.min_sample_rate();
        println!(
            "Using sample rate: {} Hz and channels: {}",
            actual_sample_rate.0,
            supported_config.channels()
        );

        let desired_config = StreamConfig {
            channels: supported_config.channels(),
            sample_rate: actual_sample_rate, // Use the supported sample rate
            buffer_size: BufferSize::Default, // Default buffer size
        };

        // Create an Arc<Mutex<Vec<f32>>> to safely share audio data across threads
        let audio_buffer = Arc::new(Queue::with_capacity(16_000 * 100));
        let audio_buffer_clone = audio_buffer.clone();

        // Create a channel to send captured audio for processing.
        let (tx, rx) = mpsc::sync_channel::<Vec<f32>>(10);

        // Set up the input audio stream.
        let tx_clone = tx.clone();
        let stream = device.build_input_stream(
            &desired_config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let audio_chunk = data.to_vec();
                if tx_clone.send(audio_chunk).is_err() {
                    eprintln!("Audio processing thread disconnected.");
                }
            },
            move |err| {
                eprintln!("An error occurred on the input stream: {}", err);
            },
            None
        )?;

        // Spawn a thread to process audio chunks.
        let processing_handle = std::thread::spawn(move || {
            // Set up the resampler.
            let mut resampler = SincFixedIn::<f32>::new(
                (16000.0 / actual_sample_rate.0 as f32).into(), // Ratio to convert to 16 kHz
                2.0, // Latency
                SincInterpolationParameters {
                    sinc_len: 256,
                    f_cutoff: 0.95,
                    interpolation: SincInterpolationType::Linear,
                    oversampling_factor: 128,
                    window: rubato::WindowFunction::Blackman,
                },
                512,
                desired_config.channels as usize,
            )
            .expect("Failed to create resampler");

            while let Ok(audio_chunk) = rx.recv() {
                if let Ok(resampled) = resampler.process(&[audio_chunk], None) {
                    for resampled_chunk in resampled {
                        process_audio(&resampled_chunk, &audio_buffer_clone);
                    }
                } else {
                    eprintln!("Error during resampling.");
                }
            }
        });

        Ok(AudioAsyncMic {
            queue: audio_buffer,
            processing_handle: Some(processing_handle),
            stream: Some(stream),
            tx: Some(tx),
        })
    }
}

impl AudioSource for AudioAsyncMic {
    fn start(&self) -> Result<(), anyhow::Error> {
        if let Some(ref stream) = self.stream {
            stream.play()?;
        } else {
            anyhow::bail!("No stream available");
        }

        Ok(())
    }

    fn stop(&mut self) -> Result<(), anyhow::Error> {
        println!("Stopping audio stream...");

        // Stop the stream
        if let Some(stream) = self.stream.take() {
            drop(stream); // Dropping the stream stops it.
        }

        // Close the channel
        if let Some(tx) = self.tx.take() {
            drop(tx); // Closing the channel.
        }

        // Join the processing thread
        if let Some(handle) = self.processing_handle.take() {
            handle.join().unwrap();
        }

        Ok(())
    }

    fn get_queue_len(&self) -> usize {
        self.queue.len()
    }

    fn drain_queue(&self) -> Result<Vec<f32>, anyhow::Error> {
        Ok(self.queue.try_dequeue()?.to_vec())
    }
}

/// Process audio chunk and store it in a shared buffer
fn process_audio(audio_chunk: &[f32], audio_buffer: &Arc<Queue<VecBuffer<f32>>>) {
    for chunk in audio_chunk.to_vec() {
        audio_buffer.try_enqueue([chunk]);
    }
}

struct AudioAsyncNew {
    capturer: Arc<Mutex<Capturer>>,
    stream: Arc<SCStream>,
    rx: Receiver<Vec<f32>>
}

impl AudioAsyncNew {
    fn new() -> Result<AudioAsyncNew, anyhow::Error> {
        let sample_rate = 16_000;
        let channel_count = 1;

        let (tx, rx) = unbounded::<Vec<f32>>();

        let config = SCStreamConfiguration::new()
            .set_captures_audio(true).unwrap()
            .set_sample_rate(sample_rate).unwrap()
            .set_channel_count(channel_count as u8).unwrap();
            // .set_capture_microphone(true).unwrap();

        let display = SCShareableContent::get().unwrap().displays().remove(0);
        let filter = SCContentFilter::new().with_display_excluding_windows(&display, &[]);
        let mut stream = SCStream::new(&filter, &config);

        let capturer = Capturer::new("asdf.wav", 16_000, 1, tx);
        let capturer_wrapper1 = CapturerWrapper { capturer: Arc::new(Mutex::new(capturer)) };
        let capturer_wrapper2 = CapturerWrapper { capturer: capturer_wrapper1.capturer.clone() };
        stream.add_output_handler(capturer_wrapper1, SCStreamOutputType::Audio);
        let stream_clone = Arc::new(stream);
        let res = AudioAsyncNew {
            capturer: capturer_wrapper2.capturer.clone(),
            stream: stream_clone,
            rx
        };


        // stream.start_capture().unwrap();

        // let ten_millis = time::Duration::from_millis(5000);

        // thread::sleep(ten_millis);

        // stream.stop_capture().unwrap();

        Ok(res)

    }
}

impl AudioSource for AudioAsyncNew {
    // fn get_queue(&self) -> &Queue<VecBuffer<f32>> {
    //     &self.capturer.lock().unwrap().queue
    // }

    // fn try_recv(&self) -> Result<Vec<f32>, crossbeam::channel::TryRecvError> {
    //     self.rx.try_recv()
    // }

    fn start(&self) -> Result<(), anyhow::Error> {
        self.stream.start_capture().unwrap();

        Ok(())
    }

    fn stop(&mut self) -> Result<(), anyhow::Error> {
        self.stream.stop_capture().unwrap();

        Ok(())
    }

    fn get_queue_len(&self) -> usize {
        self.capturer.lock().unwrap().queue.len()
    }

    fn drain_queue(&self) -> Result<Vec<f32>, anyhow::Error> {
        Ok(self.capturer.lock().unwrap().queue.try_dequeue()?.to_vec())
    }
}

struct AudioAsync {
    m_audio: Vec<f32>,
    m_audio_pos: u64,
    m_audio_len: u64,
    m_len_ms: usize,
    m_sample_rate: u64
}

impl AudioAsync {
    fn new(sample_rate: u64, len_ms: usize) -> AudioAsync {
        let audio: Vec<f32> = vec![0.0; (sample_rate as usize * len_ms) / 1000];
        AudioAsync {
            m_len_ms: len_ms,
            m_audio: audio,
            m_audio_pos: 0,
            m_audio_len: 0,
            m_sample_rate: sample_rate
        }
    }

    fn callback(&mut self, mut samples: Vec<f32>) {
        println!("samples: {:?}", samples);
        let mut n_samples = samples.len() as u64;
        let m_audio_len = self.m_audio.len() as u64;

        if n_samples > m_audio_len {
            samples.drain(0..(n_samples - m_audio_len) as usize);
            n_samples = m_audio_len;
        }

        if (self.m_audio_pos + n_samples as u64) > m_audio_len {
            let n0 = (m_audio_len - self.m_audio_pos) as usize;

            self.m_audio[(self.m_audio_pos as usize)..].copy_from_slice(&samples[..n0]);
            self.m_audio[..(n_samples as usize - n0)].copy_from_slice(&samples[n0..]);

            self.m_audio_pos = (self.m_audio_pos + n_samples as u64) % m_audio_len;
            self.m_audio_len = self.m_audio.len() as u64;
        } else {
            self.m_audio[(self.m_audio_pos as usize)..(self.m_audio_pos + n_samples) as usize].copy_from_slice(&samples);

            println!("m_audio: {:?}", self.m_audio);

            self.m_audio_pos = (self.m_audio_pos + n_samples as u64) % self.m_audio.len() as u64;
            self.m_audio_len = min(m_audio_len + n_samples, self.m_audio.len() as u64);
        }
    }

    fn get(&self, mut ms: i32) -> Vec<f32> {
        if ms <= self.m_len_ms as i32 {
            ms = self.m_len_ms as i32;
        }

        let mut n_samples = (self.m_sample_rate * ms as u64) / 1000;
        if n_samples > self.m_audio_len {
            n_samples = self.m_audio_len;
        }

        let mut result: Vec<f32> = Vec::with_capacity(n_samples as usize);

        println!("m_audio_pos: {}, n_samples: {}", self.m_audio_pos, n_samples);
        let mut s0: i32 = (self.m_audio_pos as i32 - n_samples as i32) as i32 ;
        let m_audio_len = self.m_audio.len() as u64;
        if s0 < 0 {
            s0 += m_audio_len as i32;
        }

        if (s0 + n_samples as i32) < (m_audio_len as i32) {
            let n0 = (m_audio_len as i32 - s0) as usize;

            result[..n0].copy_from_slice(&self.m_audio[s0 as usize..]);
            result[n0..].copy_from_slice(&self.m_audio[..(n_samples - n0 as u64) as usize]);
        } else {
            result.copy_from_slice(&self.m_audio[s0 as usize..(s0 + n_samples as i32) as usize])
        }

        result
    }
}

pub struct Capturer {
    stream_handle: OutputStreamHandle,
    buffer: Vec<f32>,
    tx: Sender<Vec<f32>>,
    queue: Queue<VecBuffer<f32>>,
    audio_async: AudioAsync,
    callback_count: i32
}

impl Capturer {
    pub fn new(audio_file_path: &str, sample_rate: u32, channels: u16, tx: Sender<Vec<f32>>) -> Self {
        // let audio_writer = AudioFileWriter::new(audio_file_path, sample_rate, channels);
        let (_stream, stream_handle) = OutputStream::try_default().unwrap();
        Capturer {
            stream_handle,
            buffer: Default::default(),
            tx,
            queue: Queue::with_capacity(16_000 * 100),
            audio_async: AudioAsync::new(16_000, 100_000),
            callback_count: 0
        }
    }
}

static mut AUDIO_BUFFER: Lazy<Vec<f32>> = Lazy::new(|| Default::default());

impl SCStreamOutputTrait for Capturer {
    fn did_output_sample_buffer(&self, sample: CMSampleBuffer, of_type: SCStreamOutputType) {
        // println!("[Capturer] thread id: {:?}", std::thread::current().id());
        // println!("New frame recvd");
        // println!("of_type: {:?}", of_type);
        // Assuming the audio writer is initialized and accessible
        // self.callback_count += 1;
        let binding = sample.get_audio_buffer_list().unwrap();
        let buffers = binding.buffers();
        for buf in buffers.iter() {
            // println!("number of channels: {}, data len: {:?}", buf.number_channels, buf.data.len());
            let samples: Vec<f32> = u8_to_pcmf32(&buf.data().to_vec());
            // println!("samples count: {}", samples.len());
            // self.tx.send(samples.clone()).unwrap();
            for sample in samples {
                self.queue.try_enqueue([sample]).unwrap();
            }
            // for sample in samples {
            //     self.tx.send(sample);
            // }
            // self.audio_writer.write_samples(&samples);
            // let source = SamplesBuffer::new(1, 16_000, samples);
            // self.stream_handle.play_raw(source.convert_samples()).unwrap();
            // self.buffer.append(&mut samples);
            // self.audio_async.callback(samples);
            // unsafe { AUDIO_BUFFER.append(&mut samples) };
        }
    }
}

enum AudioSourceEnum {
    Mic(AudioAsyncMic),
    New(AudioAsyncNew),
}

impl AudioSource for AudioSourceEnum {
    fn start(&self) -> Result<(), anyhow::Error> {
        match self {
            AudioSourceEnum::Mic(mic) => mic.start(),
            AudioSourceEnum::New(new) => new.start(),
        }
    }

    fn stop(&mut self) -> Result<(), anyhow::Error> {
        match self {
            AudioSourceEnum::Mic(mic) => mic.stop(),
            AudioSourceEnum::New(new) => new.stop(),
        }
    }

    fn get_queue_len(&self) -> usize {
        match self {
            AudioSourceEnum::Mic(mic) => mic.get_queue_len(),
            AudioSourceEnum::New(new) => new.get_queue_len(),
        }
    }

    fn drain_queue(&self) -> Result<Vec<f32>, anyhow::Error> {
        match self {
            AudioSourceEnum::Mic(mic) => mic.drain_queue(),
            AudioSourceEnum::New(new) => new.drain_queue(),
        }
    }
}



fn u8_to_pcmf32(data: &Vec<u8>) -> Vec<f32> {
    // Ensure the data length is a multiple of 4 since we're interpreting 4 bytes as one f32
    assert!(data.len() % 4 == 0, "Data length must be a multiple of 4.");

    // Convert chunks of 4 bytes into f32
    // Replace with: https://github.com/ardaku/fon/blob/stable/examples/mix.rs#L81
    let pcmf32: Vec<f32> = data
        .chunks_exact(4)
        .map(|chunk| {
            let bytes: [u8; 4] = chunk.try_into().expect("Chunk length should be 4 bytes.");
            f32::from_le_bytes(bytes)  // Convert little-endian bytes to f32
        })
        .collect();

    pcmf32
}

fn f32_to_i16(sample: f32) -> i16 {
    // Convert f32 sample in the range [-1.0, 1.0] to i16 in the range [-32768, 32767]
    (sample * 32767.0) as i16
}

const WHISPER_SAMPLE_RATE: i32 = 16_000;

fn transcribe(state: &mut WhisperState, speech: &mut Vec<f32>) -> String {
    let mut wparams = FullParams::new(SamplingStrategy::BeamSearch { beam_size: 5, patience: 1.0 });
    // let mut wparams = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    wparams.set_print_progress(false);
    wparams.set_print_special(false);
    wparams.set_print_realtime(false);
    wparams.set_print_timestamps(false);
    wparams.set_translate(false);
    wparams.set_single_segment(false);
    wparams.set_max_tokens(100);
    // wparams.set_no_context(true);
    // wparams.set_n_max_text_ctx(64);
    // wparams.set_audio_ctx(audio_ctx);

    wparams.set_language(Some("en"));
    wparams.set_n_threads(8);
    wparams.set_audio_ctx(0);
    wparams.set_tdrz_enable(false);
    wparams.set_temperature_inc(0.0);
    wparams.set_no_timestamps(false);
    // Remove Repetitions:
    // https://github.com/ggerganov/whisper.cpp/issues/896#issuecomment-1569586018
    // https://github.com/ggerganov/whisper.cpp/issues/471
    // https://github.com/openai/whisper/discussions/679
    wparams.set_entropy_thold(2.8);

    let mut speech2 = speech.clone();
    if speech2.len() <= WHISPER_SAMPLE_RATE as usize {
        speech2.append(&mut vec![0.0 as f32; WHISPER_SAMPLE_RATE as usize - speech2.len() + 1600])
    }

    state
        .full(wparams, &speech2)
        .expect("failed to run model");
    let num_segments = state
        .full_n_segments()
        .expect("failed to get number of segments");
    let mut text = String::new();
    for i in 0..num_segments {
        let segment = state
            .full_get_segment_text(i)
            .expect("failed to get segment");
        // if segment.len() == 0 {
        //     panic!("empty segment")
        // }
        // let num_tokens = state.full_n_tokens(i).unwrap();
        // for j in 0..num_tokens {
        //     let token = state.full_get_token_data(i, j).unwrap();
        //     // println!("t0: {}, t1: {}, dtw: {}", token.t0, token.t1, token.t_dtw);
        // }
        text.push_str(&segment);
    }

    text
}
fn end_recording(state: &mut WhisperState, speech: &mut Vec<f32>, term: &mut Term) {
    let text = transcribe(state, speech);
    print_captions(text, term, true);
}

fn print_captions(text: String, term: &mut Term, with_new_line: bool) {
    term.clear_line().unwrap();
    term.write_fmt(format_args!("{}", text)).unwrap();

    if with_new_line {
        write!(term, "\n").unwrap();
    }

    io::stdout().flush().unwrap();
}

fn main() {
    /* When more than this amount of audio received, run an iteration. */
    const trigger_ms: i32 = 400;
    const n_samples_trigger: i32 = ((trigger_ms as f32 / 1000.0) * WHISPER_SAMPLE_RATE as f32) as i32;

    /**
     * When more than this amount of audio accumulates in the audio buffer,
     * force finalize current audio context and clear the buffer. Note that
     * VAD may finalize an iteration earlier.
     */
    // This is recommended to be smaller than the time wparams.audio_ctx
    // represents so an iteration can fit in one chunk.
    const iter_threshold_ms: i32 = trigger_ms * 20;
    const n_samples_iter_threshold: i32 = ((iter_threshold_ms as f32 / 1000.0) * WHISPER_SAMPLE_RATE as f32) as i32;

    /* VAD parameters */
    // The most recent 3s.
    const vad_window_s: i32 = 3;
    const n_samples_vad_window: i32 = WHISPER_SAMPLE_RATE * vad_window_s;
    // In VAD, compare the energy of the last 500ms to that of the total 3s.
    const vad_last_ms: i32 = 500;

    // Keep the last 0.5s of an iteration to the next one for better
    // transcription at begin/end.
    const n_samples_keep_iter: i32 = (WHISPER_SAMPLE_RATE as f32 * 1.5) as i32;
    const vad_thold: f32 = 0.1;
    const freq_thold: f32 = 200.0;

    // Parse CLI arguments using the modern `clap` API
    let matches = Command::new("My Scribe")
        .version("1.0")
        .author("Alex Jiao <uohxzela@example.com>")
        .about("Live, local, and private transcription")
        .arg(
            Arg::new("source")
                .long("source")
                .short('s')
                .required(true)
                .value_parser(["mic", "sys"])
                .help("Choose the audio source: 'mic' or 'sys'"),
        )
        .get_matches();

    // Retrieve the value of the "source" argument
    let source = matches
        .get_one::<String>("source")
        .expect("Argument 'source' is required").as_str();

    // Initialize the appropriate audio source
    let mut audio: AudioSourceEnum = match source {
        "mic" => {
            println!("Using microphone audio.");
            AudioSourceEnum::Mic(AudioAsyncMic::new().expect("Failed to initialize microphone audio"))
        }
        "sys" => {
            println!("Using system audio.");
            AudioSourceEnum::New(AudioAsyncNew::new().expect("Failed to initialize system audio"))
        }
        _ => {
            eprintln!("Invalid audio source specified.");
            process::exit(1);
        }
    };

    // let audio = AudioAsyncNew::new(16_000, 1);
    // let mut audio = AudioAsyncMic::new().unwrap();
    audio.start();

    // Whisper init
    let mut whisper_ctx_params = WhisperContextParameters::default();
    whisper_ctx_params.use_gpu(true);
    whisper_ctx_params.flash_attn(true);
    // whisper_ctx_params.dtw_parameters(
    //     DtwParameters {
    //         mode: whisper_rs::DtwMode::ModelPreset { model_preset: DtwModelPreset::SmallEn },
    //         ..Default::default()
    //     }
    // );
    // whisper_ctx_params.dtw_parameters(true);

    let whisper_ctx = WhisperContext::new_with_params(
		// "/Users/jiaalex/Whisper/whisper.cpp/models/ggml-large-v3-turbo-q5_0.bin",
        // "/Users/jiaalex/Whisper/whisper.cpp/models/ggml-medium.bin",
        "/Users/jiaalex/Whisper/whisper.cpp/models/ggml-small.en.bin",
        // "/Users/jiaalex/Whisper/whisper.cpp/models/ggml-base.en.bin",
		whisper_ctx_params
	).expect("failed to load model");

    let mut state = whisper_ctx.create_state().expect("failed to create state");

    let mut pcmf32: Vec<f32> = Vec::with_capacity(WHISPER_SAMPLE_RATE as usize * 30);
    let mut pcmf32_new: Vec<f32> = Vec::with_capacity(WHISPER_SAMPLE_RATE as usize * 30);

    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();

    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    }).expect("Error setting Ctrl-C handler");

    println!("Waiting for Ctrl-C...");

    let mut term = Term::stdout();

    let mut prev_num_segments: i32 = 0;
    let mut prev_seg_len: i32 = 0;
    let mut prev_n_tokens = 0;
    let mut retry_count = 0;

    let mut silero = VoiceActivityDetector::builder()
        .sample_rate(16_000)
        .chunk_size(512usize)
        .build().unwrap();
    let mut vad = VADIterator::new(silero, 0.5, 16_000, 100, 64).unwrap();
    let mut recording = false;
    let mut recording_start_time = Instant::now();

    while running.load(Ordering::SeqCst) {
        let start_time = Instant::now();
        loop {
            let start_time = Instant::now();
            let queue_len = audio.get_queue_len();

            // let queue = &mut audio.queue;
            let duration = start_time.elapsed();
            // println!("Execution time of lock acquisition: {:?}", duration);

            if queue_len as i32 > n_samples_iter_threshold {
                println!("WARNING: cannot process audio fast enough, dropping audio ...");
                // clear audio
                audio.drain_queue().unwrap();
                continue;
            }

            if queue_len as i32 > n_samples_trigger {
                if let Ok(buffer) = audio.drain_queue() {
                    pcmf32_new.append(&mut buffer.to_vec());
                    break;
                } else {
                    // println!("alex");
                    sleep(Duration::from_millis(1));
                    continue;
                }
            }

            // println!("queue len not enough: {}", queue.len());

            // sleep(Duration::from_millis(10));
        }
        let duration_spinloop = start_time.elapsed();
        // println!("Execution time of spinloop: {:?}", duration_spinloop);
        // println!("callback count: {}", audio.capturer.lock().unwrap().callback_count);
        // audio.capturer.lock().unwrap().callback_count = 0;

        // println!("[main] thread id: {:?}", std::thread::current().id());

        pcmf32.append(&mut pcmf32_new);

        if pcmf32.len() <= WHISPER_SAMPLE_RATE as usize {
            pcmf32.append(&mut vec![0.0 as f32; WHISPER_SAMPLE_RATE as usize - pcmf32.len() + 1600])
        }

        let mut wparams = FullParams::new(SamplingStrategy::BeamSearch { beam_size: 5, patience: 1.0 });
        // let mut wparams = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        wparams.set_print_progress(false);
        wparams.set_print_special(false);
        wparams.set_print_realtime(false);
        wparams.set_print_timestamps(false);
        wparams.set_translate(false);
        wparams.set_single_segment(false);
        wparams.set_max_tokens(100);
        // wparams.set_no_context(true);
        // wparams.set_n_max_text_ctx(64);
        // wparams.set_audio_ctx(audio_ctx);

        wparams.set_language(Some("en"));
        wparams.set_n_threads(8);
        wparams.set_audio_ctx(0);
        wparams.set_tdrz_enable(false);
        wparams.set_temperature_inc(0.0);
        wparams.set_no_timestamps(false);
        // Remove Repetitions:
        // https://github.com/ggerganov/whisper.cpp/issues/896#issuecomment-1569586018
        // https://github.com/ggerganov/whisper.cpp/issues/471
        // https://github.com/openai/whisper/discussions/679
        wparams.set_entropy_thold(2.8);
        // wparams.set_logprob_thold(-2.0);
        // wparams.set_no_speech_thold(0.8);
        // wparams.set_token_timestamps(true);
        // wparams.set_split_on_word(true);
        // wparams.set_max_len(1);
        // wparams.set_logprob_thold(-10.0);
        // wparams.set_max_len(10);
        // wparams.set_no_speech_thold(0.5);
        // wparams.set_suppress_blank(true);

        // sleep(Duration::from_secs(2));

        // continue;

        // now we can run the model
        let data = &pcmf32.clone()[..];
        // println!("transcribing {} samples", data.len());

        let mut speech: Vec<f32> = Vec::new();
        let lookback_size = 5 * 512;

        let vad_start = Instant::now();

        let vad = VoiceActivityDetector::builder()
            .sample_rate(16000)
            .chunk_size(512usize)
            .build().unwrap();

        // This will label any audio chunks with a probability greater than 75% as speech,
        // and label the 3 additional chunks before and after these chunks as speech.
        let labels = pcmf32.clone().into_iter().label(vad, 0.75, 5);
        // let num_labels = labels.count();
        let mut filtered_samples: Vec<f32> = Vec::new();
        for (i, label) in labels.enumerate() {
            match label {
                LabeledAudio::Speech(chunk) => {
                    filtered_samples.extend(chunk);
                },
                LabeledAudio::NonSpeech(_) => {
                    // println!("non-speech detected!")
                }
            }
        }

        if filtered_samples.len() <= WHISPER_SAMPLE_RATE as usize {
            filtered_samples.append(&mut vec![0.0 as f32; WHISPER_SAMPLE_RATE as usize - filtered_samples.len() + 1600])
        }

        // println!("elapsed vad time: {:?}", vad_start.elapsed());

        // for chunk in data.chunks_exact(512) {
        //     speech.extend(chunk);
        //     if !recording && speech.len() > lookback_size {
        //         speech = speech[speech.len() - lookback_size..].to_vec();
        //     }
        //     let vad_result = vad.process(chunk.to_vec());
        //     match vad_result {
        //         VADResult::Start(_) => {
        //             if !recording {
        //                 println!("start");
        //                 recording = true;
        //                 recording_start_time = Instant::now();
        //             }
        //         },
        //         VADResult::End(_) => {
        //             if recording {
        //                 println!("end");
        //                 recording = false;
        //                 end_recording(&mut state, &mut speech, &mut term);
        //                 pcmf32.clear();
        //             }
        //         },
        //         VADResult::None => {
        //             if recording {
        //                 if speech.len() / WHISPER_SAMPLE_RATE as usize > 8 {
        //                     recording = false;
        //                     end_recording(&mut state, &mut speech, &mut term);
        //                     pcmf32.clear();
        //                     vad.soft_reset();
        //                 }

        //                 if recording_start_time.elapsed() > Duration::from_millis(400) {

        //                     let text = transcribe(&mut state, &mut speech);
        //                     println!("refresh: {}", text);
        //                     print_captions(text, &mut term, false);
        //                     recording_start_time = Instant::now();
        //                 }
        //             }
        //         }
        //     };
        // }

        // continue;

        state
            .full(wparams, &filtered_samples)
            .expect("failed to run model");
        let duration_full = start_time.elapsed();
        // println!("Execution time of full_whisper: {:?}", duration_full);

        // let mut stdout = std::io::stdout().into_raw_mode().unwrap();
        // let (row, col) = stdout.cursor_pos().unwrap();
        // print!("{}{}", termion::clear::CurrentLine, termion::cursor::Goto(col, row));

        // if let Some(num) = prev_num_segments {
        //     if num > 0 {

        //     }
        // }
        // let color: impl Fn(i32) -> termion::color::Fg<U : termion::color::Color> = |i: i32| {
        //     match i {
        //         0 => termion::color::Fg(termion::color::LightCyan),
        //         1 => termion::color::Fg(termion::color::LightGreen),
        //         2 => termion::color::LightMagenta,
        //         3 => termion::color::LightYellow,
        //     }
        // };
        // fetch the results
        let num_segments = state
            .full_n_segments()
            .expect("failed to get number of segments");

        let mut segment_len = 0;
        let mut n_tokens = 0;
        for i in 0..num_segments {
            segment_len += state.full_get_segment_text(i).unwrap().len() as i32;
            n_tokens += state.full_n_tokens(i).unwrap();

        }

        prev_n_tokens = n_tokens;

        // if n_tokens < prev_n_tokens {
        //     continue;
        // }

        // if (n_tokens - prev_n_tokens) < 32 && (n_tokens - prev_n_tokens) > 0 {
        //     prev_n_tokens = n_tokens;
        // } else {
        //     continue;
        // }

        if segment_len < prev_seg_len && retry_count < 5 {
            retry_count += 1;
            // Mostly likely endless repetition here
            continue;
        }

        retry_count = 0;
        prev_seg_len = segment_len;

        if num_segments == 1 && num_segments < prev_num_segments {
            // println!("num_segments < prev_num_segments");
            // write!(term, "curr: {}, prev: {}, {:?}", num_segments, prev_num_segments, duration_full).unwrap();

            // continue;
        }
        prev_num_segments = num_segments;
        if num_segments > 0 {
            term.clear_line().unwrap();
        } else {
            // panic!("no segment")
            continue;
        }

        // let mut segment_text = String::new();
        for i in 0..num_segments {
            let segment = state
                .full_get_segment_text(i)
                .expect("failed to get segment");
            // if segment.len() == 0 {
            //     panic!("empty segment")
            // }
            // let num_tokens = state.full_n_tokens(i).unwrap();
            // for j in 0..num_tokens {
            //     let token = state.full_get_token_data(i, j).unwrap();
            //     // println!("t0: {}, t1: {}, dtw: {}", token.t0, token.t1, token.t_dtw);
            // }

            term.write_fmt(format_args!("{}", segment)).unwrap();


            io::stdout().flush().unwrap();
        }

        let mut speech_has_end = false;

        /* Need enough accumulated audio to do VAD. */
        if pcmf32.len() >= n_samples_vad_window as usize {
            let pcmf32_window: Vec<f32> = pcmf32[pcmf32.len() - (n_samples_vad_window as usize)..].to_vec();

            let start = Instant::now();
            speech_has_end = vad_simple(&mut pcmf32_window.clone(), WHISPER_SAMPLE_RATE, vad_last_ms, vad_thold, freq_thold, false);
            let elapsed = start.elapsed();

            // println!("Execution time of vad_simple: {:.3} ms", elapsed.as_secs_f64() * 1000.0);

            if speech_has_end {
                // println!("\nspeech end detected\n");
            }
        }

        if pcmf32.len() as i32 > n_samples_iter_threshold || speech_has_end {
            // Don't terminate current line if curr n_tokens below a certain threshold
            if segment_len < 15 {
                // If this condition is hit then there's no speech for 15+ secs
                if pcmf32.len() as i32 > n_samples_iter_threshold {
                    pcmf32.clear();
                    continue;
                }
                continue;
            }

            prev_num_segments = 0;
            prev_seg_len = 0;
            prev_n_tokens = 0;
            write!(term, "\n").unwrap();

            let index: i32 = pcmf32.len() as i32 - n_samples_keep_iter;
            let last: Vec<f32> = pcmf32[index as usize..].to_vec();
            if speech_has_end {
                pcmf32.clear();
            } else {
                pcmf32 = last;
            }
            // pcmf32 = last;
        }

        io::stdout().flush().unwrap();
    }
    audio.stop();
    println!("Got it! Exiting...");
}

use std::cmp;

#[derive(Debug)]
struct TranscriptionBuffer {
    current_text: String,
    prev_segment: String,
}

impl TranscriptionBuffer {
    fn new() -> Self {
        Self {
            current_text: String::new(),
            prev_segment: String::new(),
        }
    }

    fn join_segments(&mut self, new_segment: &str) -> String {
        if self.prev_segment.is_empty() {
            self.prev_segment = new_segment.to_string();
            return new_segment.to_string();
        }

        // Find the overlap point using sliding window and edit distance
        let overlap_point = self.find_overlap_point(&self.prev_segment, new_segment);

        if let Some((start_idx, _)) = overlap_point {
            // Join segments at the found overlap point
            let joined = format!("{}{}",
                self.prev_segment,
                &new_segment[start_idx..]
            );
            self.prev_segment = joined.clone();
            joined
        } else {
            // If no overlap found, just append with a space
            let joined = format!("{} {}", self.prev_segment, new_segment);
            self.prev_segment = joined.clone();
            joined
        }
    }

    fn find_overlap_point(&self, prev: &str, current: &str) -> Option<(usize, f32)> {
        let prev_words: Vec<&str> = prev.split_whitespace().collect();
        let curr_words: Vec<&str> = current.split_whitespace().collect();

        // Look for overlapping sequences
        let min_overlap = 3; // Minimum words to consider as valid overlap
        let max_overlap = cmp::min(prev_words.len(), curr_words.len());

        let mut best_match: Option<(usize, f32)> = None;
        let mut min_distance = f32::MAX;

        for window_size in (min_overlap..=max_overlap).rev() {
            for start_idx in 0..curr_words.len() - window_size + 1 {
                if start_idx + window_size > curr_words.len() {
                    continue;
                }

                let curr_window = &curr_words[start_idx..start_idx + window_size];

                // Look for this window in the previous segment
                for prev_start in 0..=prev_words.len() - window_size {
                    let prev_window = &prev_words[prev_start..prev_start + window_size];

                    let distance = self.compute_window_distance(prev_window, curr_window);

                    // If we found a perfect match
                    if distance == 0.0 {
                        let char_pos = curr_words[..start_idx].iter()
                            .map(|w| w.len() + 1)
                            .sum();
                        return Some((char_pos, 0.0));
                    }

                    // Keep track of best partial match
                    if distance < min_distance {
                        min_distance = distance;
                        let char_pos = curr_words[..start_idx].iter()
                            .map(|w| w.len() + 1)
                            .sum();
                        best_match = Some((char_pos, distance));
                    }
                }
            }
        }

        // Return best match if it's good enough
        if min_distance < 0.3 { // Threshold for acceptable partial matches
            best_match
        } else {
            None
        }
    }

    fn compute_window_distance(&self, window1: &[&str], window2: &[&str]) -> f32 {
        if window1.len() != window2.len() {
            return f32::MAX;
        }

        let total_words = window1.len();
        let mut total_distance = 0.0;

        for (w1, w2) in window1.iter().zip(window2.iter()) {
            total_distance += self.levenshtein_distance(w1, w2) as f32;
        }

        total_distance / total_words as f32
    }

    fn levenshtein_distance(&self, s1: &str, s2: &str) -> usize {
        let len1 = s1.chars().count();
        let len2 = s2.chars().count();

        if len1 == 0 { return len2; }
        if len2 == 0 { return len1; }

        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

        for i in 0..=len1 {
            matrix[i][0] = i;
        }
        for j in 0..=len2 {
            matrix[0][j] = j;
        }

        for (i, c1) in s1.chars().enumerate() {
            for (j, c2) in s2.chars().enumerate() {
                let substitution_cost = if c1 == c2 { 0 } else { 1 };
                matrix[i + 1][j + 1] = [
                    matrix[i][j + 1] + 1,                // deletion
                    matrix[i + 1][j] + 1,                // insertion
                    matrix[i][j] + substitution_cost,    // substitution
                ].iter().min().unwrap().clone();
            }
        }

        matrix[len1][len2]
    }

    fn clear_on_speech_end(&mut self) {
        self.prev_segment.clear();
    }
}

fn high_pass_filter(data: &mut [f32], cutoff: f32, sample_rate: f32) {
    const PI: f32 = std::f32::consts::PI;
    let rc = 1.0 / (2.0 * PI * cutoff);
    let dt = 1.0 / sample_rate;
    let alpha = dt / (rc + dt);

    let mut y = data[0];

    for i in 1..data.len() {
        y = alpha * (y + data[i] - data[i - 1]);
        data[i] = y;
    }
}

fn vad_simple(pcmf32: &mut [f32], sample_rate: i32, last_ms: i32, vad_thold: f32, freq_thold: f32, verbose: bool) -> bool {
    let n_samples = pcmf32.len();
    let n_samples_last = (sample_rate * last_ms) as usize / 1000;

    if n_samples_last >= n_samples {
        // not enough samples - assume no speech
        return false;
    }

    if freq_thold > 0.0 {
        high_pass_filter(pcmf32, freq_thold, sample_rate as f32);
    }

    let mut energy_all = 0.0;
    let mut energy_last = 0.0;

    for (i, &sample) in pcmf32.iter().enumerate() {
        energy_all += sample.abs();

        if i >= n_samples - n_samples_last {
            energy_last += sample.abs();
        }
    }

    energy_all /= n_samples as f32;
    energy_last /= n_samples_last as f32;

    if verbose {
        eprintln!(
            "vad_simple: energy_all: {}, energy_last: {}, vad_thold: {}, freq_thold: {}",
            energy_all, energy_last, vad_thold, freq_thold
        );
    }

    energy_last <= vad_thold * energy_all
}

/// Enum to represent the output of the VADIterator
#[derive(Debug, PartialEq)]
pub enum VADResult {
    Start(usize), // Start index in samples
    End(usize),   // End index in samples
    None,         // No event
}

pub struct VADIterator {
    model: VoiceActivityDetector,
    threshold: f32,
    sampling_rate: usize,
    min_silence_samples: usize,
    speech_pad_samples: usize,
    triggered: bool,
    temp_end: usize,
    current_sample: usize,
}

impl VADIterator {
    pub fn new(
        model: VoiceActivityDetector,
        threshold: f32,
        sampling_rate: usize,
        min_silence_duration_ms: usize,
        speech_pad_ms: usize,
    ) -> Result<Self, Error> {
        if sampling_rate != 8000 && sampling_rate != 16000 {
            return Err(Error::VadConfigError { sample_rate: 16_000, chunk_size: 512 });
        }

        let min_silence_samples = sampling_rate * min_silence_duration_ms / 1000;
        let speech_pad_samples = sampling_rate * speech_pad_ms / 1000;

        Ok(Self {
            model,
            threshold,
            sampling_rate,
            min_silence_samples,
            speech_pad_samples,
            triggered: false,
            temp_end: 0,
            current_sample: 0,
        })
    }

    pub fn soft_reset(&mut self) {
        self.triggered = false;
        self.temp_end = 0;
        self.current_sample = 0;
    }

    pub fn reset_states(&mut self) {
        self.model.reset();
        self.triggered = false;
        self.temp_end = 0;
        self.current_sample = 0;
    }

    pub fn process(&mut self, audio_chunk: Vec<f32>) -> VADResult {
        let window_size_samples = audio_chunk.len();
        self.current_sample += window_size_samples;

        let speech_prob = self.model.predict(audio_chunk);

        if speech_prob >= self.threshold && self.temp_end != 0 {
            self.temp_end = 0;
        }

        if speech_prob >= self.threshold && !self.triggered {
            self.triggered = true;
            let speech_start = self
                .current_sample
                .saturating_sub(self.speech_pad_samples + window_size_samples);
            return VADResult::Start(speech_start);
        }

        if speech_prob < self.threshold - 0.15 && self.triggered {
            if self.temp_end == 0 {
                self.temp_end = self.current_sample;
            }
            if self.current_sample - self.temp_end < self.min_silence_samples {
                return VADResult::None;
            } else {
                let speech_end = self.temp_end + self.speech_pad_samples - window_size_samples;
                self.temp_end = 0;
                self.triggered = false;
                return VADResult::End(speech_end);
            }
        }

        VADResult::None
    }
}

// fn main2() {
//     println!("Starting");

//     let content = SCShareableContent::current();
//     let displays = content.displays;

//     let display = displays.first().unwrap_or_else(|| {
//         panic!("Main display not found");
//     });
//     let display = display.to_owned();

//     let width = display.width;
//     let height = display.height;

//     let params = InitParams::Display(display);
//     let filter = SCContentFilter::new(params);

//     let stream_config = SCStreamConfiguration {
//         width,
//         height,
//         captures_audio: true,
//         sample_rate: 16_000,
//         channel_count: 1,
//         ..Default::default()
//     };

//     let (tx, rx) = unbounded::<Vec<f32>>();

//     let mut stream = SCStream::new(filter, stream_config, ErrorHandler);
//     let capturer = Capturer::new("asdf.wav", 16_000, 1, tx);
//     let capturer_wrapper1 = CapturerWrapper { capturer: Arc::new(Mutex::new(capturer)) };
//     let capturer_wrapper2 = CapturerWrapper { capturer: capturer_wrapper1.capturer.clone() };
//     stream.add_output(capturer_wrapper1, SCStreamOutputType::Audio);

//     stream.start_capture().unwrap();

//     let ten_millis = time::Duration::from_millis(5000);

//     thread::sleep(ten_millis);

//     stream.stop_capture().unwrap();

//     let mut whisper_ctx_params = WhisperContextParameters::default();
//     whisper_ctx_params.use_gpu(true);

//     let ctx = WhisperContext::new_with_params(
// 		"/Users/jiaalex/Whisper/whisper.cpp/models/ggml-small.en.bin",
// 		whisper_ctx_params
// 	).expect("failed to load model");

//     	// create a params object
// 	let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

// 	// assume we have a buffer of audio data
// 	// here we'll make a fake one, floating point samples, 32 bit, 16KHz, mono
// 	// let audio_data = &capturer_wrapper2.capturer.lock().unwrap().audio_async.get(-1);
//     let capturer = &capturer_wrapper2.capturer.lock().unwrap();

//     let audio_data = capturer.queue.try_dequeue().unwrap();

// 	// now we can run the model
// 	let mut state = ctx.create_state().expect("failed to create state");
// 	state
// 		.full(params, &audio_data[..])
// 		.expect("failed to run model");

// 	// fetch the results
// 	let num_segments = state
// 		.full_n_segments()
// 		.expect("failed to get number of segments");
// 	for i in 0..num_segments {
// 		let segment = state
// 			.full_get_segment_text(i)
// 			.expect("failed to get segment");
// 		let start_timestamp = state
// 			.full_get_segment_t0(i)
// 			.expect("failed to get segment start timestamp");
// 		let end_timestamp = state
// 			.full_get_segment_t1(i)
// 			.expect("failed to get segment end timestamp");
// 		println!("[{} - {}]: {}", start_timestamp, end_timestamp, segment);
// 	}

//     println!("Ended");
// }
