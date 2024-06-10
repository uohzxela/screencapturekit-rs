use core::num;
use std::{cmp::min, fs::File, io::{self, BufWriter, Write}, path::PathBuf, sync::{atomic::{AtomicBool, Ordering}, Arc, Mutex}, thread::{self, sleep}, time::{self, Duration, Instant}};

use console::Term;
use cpal::{traits::{HostTrait, StreamTrait}, FromSample, Sample, StreamConfig};
use hound::{WavSpec, WavWriter};
use rodio::{buffer::SamplesBuffer, DeviceTrait, OutputStream, OutputStreamHandle, Source};
use screencapturekit::{
    cm_sample_buffer::CMSampleBuffer,
    sc_content_filter::{InitParams, SCContentFilter},
    sc_error_handler::StreamErrorHandler,
    sc_output_handler::{SCStreamOutputType, StreamOutput},
    sc_shareable_content::SCShareableContent,
    sc_stream::SCStream,
    sc_stream_configuration::SCStreamConfiguration,
};
use screencapturekit_sys::os_types::geometry::{CGPoint, CGRect, CGSize};
use swap_buffer_queue::{buffer::VecBuffer, Queue};
use termion::{cursor::DetectCursorPos, raw::IntoRawMode};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};
use once_cell::sync::Lazy;
use crossbeam::channel::{unbounded, Receiver, Sender};

struct ErrorHandler;
impl StreamErrorHandler for ErrorHandler {
    fn on_error(&self) {
        println!("Error!");
    }
}

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

impl StreamOutput for CapturerWrapper {
    fn did_output_sample_buffer(&mut self, sample_buffer: CMSampleBuffer, of_type: SCStreamOutputType) {
        let start_time = Instant::now();
        self.capturer.lock().unwrap().did_output_sample_buffer(sample_buffer, of_type);
        let duration = start_time.elapsed();
        // println!("Execution time: {:?}", duration);
    }
}

struct AudioAsyncMic {

}

impl AudioAsyncMic {
    fn new() {
        let host = cpal::default_host();

        let device = host.default_input_device().expect("failed to find input device");

        println!("Input device: {}", device.name().expect("failed to find input device name"));

        let config = device
            .default_input_config()
            .expect("Failed to get default input config");

        let err_fn = move |err| {
            eprintln!("an error occurred on stream: {}", err);
        };

        println!("Input config: {:?}", config);

        let stream = device.build_input_stream(
            &config.into(),
            move |data, _: &_| {
                println!("hello");
                write_input_data(data)
            },
            err_fn,
            None
        ).expect("Failed to build input stream");

        stream.play().expect("Failed to play stream");

        std::thread::sleep(std::time::Duration::from_secs(3));
        // drop(stream);
    }

    fn new2() {
        let sdl_context = sdl2::init().unwrap();
    }
}

fn write_input_data(input: &[f32])
{
    println!("samples from cpal: {:?}", input.len());
}

struct AudioAsyncNew {
    capturer: Arc<Mutex<Capturer>>,
    stream: Arc<SCStream>,
    rx: Receiver<Vec<f32>>
}

impl AudioAsyncNew {
    fn new(sample_rate: u32, channel_count: u32) -> AudioAsyncNew {
        let content = SCShareableContent::current();
        let displays = content.displays;

        let display = displays.first().unwrap_or_else(|| {
            panic!("Main display not found");
        });
        let display = display.to_owned();

        let params = InitParams::Display(display);
        let filter = SCContentFilter::new(params);

        let stream_config = SCStreamConfiguration {
            width: 1,
            height: 1,
            captures_audio: true,
            sample_rate,
            channel_count,
            ..Default::default()
        };

        let (tx, rx) = unbounded::<Vec<f32>>();

        let mut stream = SCStream::new(filter, stream_config, ErrorHandler);
        let capturer = Capturer::new("asdf.wav", 16_000, 1, tx);
        let capturer_wrapper1 = CapturerWrapper { capturer: Arc::new(Mutex::new(capturer)) };
        let capturer_wrapper2 = CapturerWrapper { capturer: capturer_wrapper1.capturer.clone() };
        stream.add_output(capturer_wrapper1, SCStreamOutputType::Audio);
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

        res

    }

    // fn get_queue(&self) -> &Queue<VecBuffer<f32>> {
    //     &self.capturer.lock().unwrap().queue
    // }

    fn try_recv(&self) -> Result<Vec<f32>, crossbeam::channel::TryRecvError> {
        self.rx.try_recv()
    }

    fn start(&self) {
        self.stream.start_capture().unwrap();
    }

    fn stop(&self) {
        self.stream.stop_capture().unwrap();
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

impl StreamErrorHandler for Capturer {
    fn on_error(&self) {
        eprintln!("ERROR!");
    }
}

static mut AUDIO_BUFFER: Lazy<Vec<f32>> = Lazy::new(|| Default::default());

impl StreamOutput for Capturer {
    fn did_output_sample_buffer(&mut self, sample: CMSampleBuffer, of_type: SCStreamOutputType) {
        // println!("[Capturer] thread id: {:?}", std::thread::current().id());
        // println!("New frame recvd");
        // println!("of_type: {:?}", of_type);
        // Assuming the audio writer is initialized and accessible
        self.callback_count += 1;
        let buffers = sample.sys_ref.get_av_audio_buffer_list();
        for buf in buffers.iter() {
            // println!("number of channels: {}, data len: {:?}", buf.number_channels, buf.data.len());
            let samples: Vec<f32> = u8_to_pcmf32(&buf.data);
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

fn main() {
    AudioAsyncMic::new2();

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
    const n_samples_keep_iter: i32 = (WHISPER_SAMPLE_RATE as f32 * 0.5) as i32;
    const vad_thold: f32 = 0.3;
    const freq_thold: f32 = 200.0;

    let audio = AudioAsyncNew::new(16_000, 1);
    audio.start();

    // Whisper init
    let mut whisper_ctx_params = WhisperContextParameters::default();
    whisper_ctx_params.use_gpu(true);
    whisper_ctx_params.flash_attn(true);

    let whisper_ctx = WhisperContext::new_with_params(
		"/Users/jiaalex/Whisper/whisper.cpp/models/ggml-small.en.bin",
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

    while running.load(Ordering::SeqCst) {
        let start_time = Instant::now();
        loop {
            // while let Ok(mut samples) = audio.try_recv() {
            //     pcmf32_new.append(&mut samples);
            //     // if pcmf32_new.len() as i32 > n_samples_trigger {
            //     //     break;
            //     // }
            // }

            // if pcmf32_new.len() as i32 > n_samples_iter_threshold {
            //     println!("WARNING: cannot process audio fast enough, dropping audio ...");
            //     // clear audio
            //     pcmf32_new.clear();
            //     continue;
            // }

            // if pcmf32_new.len() as i32 > n_samples_trigger {
            //     break;
            // }

            let start_time = Instant::now();
            let queue = &audio.capturer.lock().unwrap().queue;
            let duration = start_time.elapsed();
            // println!("Execution time of lock acquisition: {:?}", duration);

            if queue.len() as i32 > n_samples_iter_threshold {
                println!("WARNING: cannot process audio fast enough, dropping audio ...");
                // clear audio
                drop(queue.try_dequeue().unwrap());
                continue;
            }

            if queue.len() as i32 > n_samples_trigger {
                if let Ok(buffer) = queue.try_dequeue() {
                    pcmf32_new.append(&mut buffer.to_vec());
                    break;
                } else {
                    println!("alex");
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
        audio.capturer.lock().unwrap().callback_count = 0;

        // println!("[main] thread id: {:?}", std::thread::current().id());

        pcmf32.append(&mut pcmf32_new);

        if pcmf32.len() <= WHISPER_SAMPLE_RATE as usize {
            pcmf32.append(&mut vec![0.0 as f32; WHISPER_SAMPLE_RATE as usize - pcmf32.len() + 1600])
        }

        // let mut wparams = FullParams::new(SamplingStrategy::BeamSearch { beam_size: 5, patience: 0.0 });
        let mut wparams = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        wparams.set_print_progress(false);
        wparams.set_print_special(false);
        wparams.set_print_realtime(false);
        wparams.set_print_timestamps(false);
        wparams.set_translate(false);
        wparams.set_single_segment(true);
        wparams.set_max_tokens(0);
        wparams.set_language(Some("en"));
        wparams.set_n_threads(8);
        wparams.set_audio_ctx(0);
        wparams.set_tdrz_enable(false);
        wparams.set_temperature_inc(0.0);
        wparams.set_no_timestamps(true);
        // wparams.set_logprob_thold(-10.0);
        // wparams.set_max_len(10);
        // wparams.set_no_speech_thold(0.2);
        // wparams.set_suppress_blank(false);

        let start_time = Instant::now();

        // sleep(Duration::from_secs(2));

        // continue;

        // now we can run the model
        let data = &pcmf32.clone()[..];
        // println!("transcribing {} samples", data.len());
        state
            .full(wparams, data)
            .expect("failed to run model");

        let duration_full = start_time.elapsed();
        // println!("Execution time of full_whisper: {:?}", duration);

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

        // let mut segment_len = 0;
        // let mut n_tokens = 0;
        // for i in 0..num_segments {
        //     segment_len += state.full_get_segment_text(i).unwrap().len() as i32;
        //     n_tokens += state.full_n_tokens(i).unwrap();

        // }

        // if n_tokens < prev_n_tokens {
        //     continue;
        // }

        // if (n_tokens - prev_n_tokens) < 20 {
        //     prev_n_tokens = n_tokens;
        // }


        // if segment_len < prev_seg_len {
        //     continue;
        // }
        // prev_seg_len = segment_len;

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

        for i in 0..num_segments {
            let segment = state
                .full_get_segment_text(i)
                .expect("failed to get segment");
            if segment.len() == 0 {
                panic!("empty segment")
            }
            // let num_tokens = state.full_n_tokens(i).unwrap();
            // for j in 0..num_tokens {
            //     let prob = state.full_get_token_prob(i, j).unwrap();
            //     println!("prob: {}", prob);
            // }

            // match i {
            //     0 => term.write_fmt(format_args!("{}{}", segment, termion::color::Fg(termion::color::Green))).unwrap(),
            //     1 => term.write_fmt(format_args!("{}{}", segment, termion::color::Fg(termion::color::Cyan))).unwrap(),
            //     2 => term.write_fmt(format_args!("{}{}", segment, termion::color::Fg(termion::color::Yellow))).unwrap(),
            //     3 => term.write_fmt(format_args!("{}{}", segment, termion::color::Fg(termion::color::Magenta))).unwrap(),
            //     _ => term.write_fmt(format_args!("{}{}", segment, termion::color::Fg(termion::color::Green))).unwrap()
            // }

            // term.write_fmt(format_args!("[{}]{}", i, segment)).unwrap();

            term.write_fmt(format_args!("{}", segment)).unwrap();


            io::stdout().flush().unwrap();

            // let start_timestamp = state
            //     .full_get_segment_t0(i)
            //     .expect("failed to get segment start timestamp");
            // let end_timestamp = state
            //     .full_get_segment_t1(i)
            //     .expect("failed to get segment end timestamp");
            // println!("[{} - {}]: {}", start_timestamp, end_timestamp, segment);
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
            // pcmf32.clear();
            prev_num_segments = 0;
            prev_seg_len = 0;
            prev_n_tokens = 0;
            // write!(term, " ({:?})", duration_spinloop).unwrap();
            // write!(term, " ({:?})", num_segments).unwrap();
            // write!(term, " ({:?})", n_tokens).unwrap();
            write!(term, "\n").unwrap();

            let index: i32 = pcmf32.len() as i32 - n_samples_keep_iter;
            let last: Vec<f32> = pcmf32[index as usize..].to_vec();
            pcmf32 = last;
        }

        io::stdout().flush().unwrap();

    }
    println!("Got it! Exiting...");
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

fn main2() {
    println!("Starting");

    let content = SCShareableContent::current();
    let displays = content.displays;

    let display = displays.first().unwrap_or_else(|| {
        panic!("Main display not found");
    });
    let display = display.to_owned();

    let width = display.width;
    let height = display.height;

    let params = InitParams::Display(display);
    let filter = SCContentFilter::new(params);

    let stream_config = SCStreamConfiguration {
        width,
        height,
        captures_audio: true,
        sample_rate: 16_000,
        channel_count: 1,
        ..Default::default()
    };

    let (tx, rx) = unbounded::<Vec<f32>>();

    let mut stream = SCStream::new(filter, stream_config, ErrorHandler);
    let capturer = Capturer::new("asdf.wav", 16_000, 1, tx);
    let capturer_wrapper1 = CapturerWrapper { capturer: Arc::new(Mutex::new(capturer)) };
    let capturer_wrapper2 = CapturerWrapper { capturer: capturer_wrapper1.capturer.clone() };
    stream.add_output(capturer_wrapper1, SCStreamOutputType::Audio);

    stream.start_capture().unwrap();

    let ten_millis = time::Duration::from_millis(5000);

    thread::sleep(ten_millis);

    stream.stop_capture().unwrap();

    let mut whisper_ctx_params = WhisperContextParameters::default();
    whisper_ctx_params.use_gpu(true);

    let ctx = WhisperContext::new_with_params(
		"/Users/jiaalex/Whisper/whisper.cpp/models/ggml-small.en.bin",
		whisper_ctx_params
	).expect("failed to load model");

    	// create a params object
	let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

	// assume we have a buffer of audio data
	// here we'll make a fake one, floating point samples, 32 bit, 16KHz, mono
	// let audio_data = &capturer_wrapper2.capturer.lock().unwrap().audio_async.get(-1);
    let capturer = &capturer_wrapper2.capturer.lock().unwrap();

    let audio_data = capturer.queue.try_dequeue().unwrap();

	// now we can run the model
	let mut state = ctx.create_state().expect("failed to create state");
	state
		.full(params, &audio_data[..])
		.expect("failed to run model");

	// fetch the results
	let num_segments = state
		.full_n_segments()
		.expect("failed to get number of segments");
	for i in 0..num_segments {
		let segment = state
			.full_get_segment_text(i)
			.expect("failed to get segment");
		let start_timestamp = state
			.full_get_segment_t0(i)
			.expect("failed to get segment start timestamp");
		let end_timestamp = state
			.full_get_segment_t1(i)
			.expect("failed to get segment end timestamp");
		println!("[{} - {}]: {}", start_timestamp, end_timestamp, segment);
	}

    println!("Ended");
}
