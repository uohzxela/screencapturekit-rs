use std::{fs::File, io::BufWriter, path::PathBuf, sync::{Arc, Mutex}, thread, time};

use hound::{WavSpec, WavWriter};
use rodio::{buffer::SamplesBuffer, OutputStream, OutputStreamHandle, Source};
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
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};
use once_cell::sync::Lazy;
use crossbeam::channel::{unbounded, Sender};

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
        self.capturer.lock().unwrap().did_output_sample_buffer(sample_buffer, of_type)
    }
}

pub struct Capturer {
    audio_writer: AudioFileWriter,
    stream_handle: OutputStreamHandle,
    buffer: Vec<f32>,
    tx: Sender<f32>
}

impl Capturer {
    pub fn new(audio_file_path: &str, sample_rate: u32, channels: u16, tx: Sender<f32>) -> Self {
        let audio_writer = AudioFileWriter::new(audio_file_path, sample_rate, channels);
        let (_stream, stream_handle) = OutputStream::try_default().unwrap();
        Capturer {
            audio_writer,
            stream_handle,
            buffer: Default::default(),
            tx
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
        println!("New frame recvd");
        println!("of_type: {:?}", of_type);
        // Assuming the audio writer is initialized and accessible

        let buffers = sample.sys_ref.get_av_audio_buffer_list();
        for buf in buffers.iter() {
            println!("number of channels: {}, data len: {:?}", buf.number_channels, buf.data.len());
            let mut samples: Vec<f32> = u8_to_pcmf32(&buf.data);
            // for sample in samples {
            //     self.tx.send(sample);
            // }
            // self.audio_writer.write_samples(&samples);
            // let source = SamplesBuffer::new(1, 16_000, samples);
            // self.stream_handle.play_raw(source.convert_samples()).unwrap();
            self.buffer.append(&mut samples);
            // unsafe { AUDIO_BUFFER.append(&mut samples) };
        }
    }
}


fn u8_to_pcmf32(data: &Vec<u8>) -> Vec<f32> {
    // Ensure the data length is a multiple of 4 since we're interpreting 4 bytes as one f32
    assert!(data.len() % 4 == 0, "Data length must be a multiple of 4.");

    // Convert chunks of 4 bytes into f32
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


fn main() {
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

    let (tx, rx) = unbounded::<f32>();

    let mut stream = SCStream::new(filter, stream_config, ErrorHandler);
    let capturer = Capturer::new("asdf.wav", 16_000, 1, tx);
    let capturer_wrapper1 = CapturerWrapper { capturer: Arc::new(Mutex::new(capturer)) };
    let capturer_wrapper2 = CapturerWrapper { capturer: capturer_wrapper1.capturer.clone() };
    stream.add_output(capturer_wrapper1, SCStreamOutputType::Audio);

    stream.start_capture().unwrap();

    let ten_millis = time::Duration::from_millis(5000);

    thread::sleep(ten_millis);

    stream.stop_capture().unwrap();

    let ctx = WhisperContext::new_with_params(
		"/Users/jiaalex/Whisper/whisper.cpp/models/ggml-small.en.bin",
		WhisperContextParameters::default()
	).expect("failed to load model");

    	// create a params object
	let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

	// assume we have a buffer of audio data
	// here we'll make a fake one, floating point samples, 32 bit, 16KHz, mono
	let audio_data = &capturer_wrapper2.capturer.lock().unwrap().buffer;

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
