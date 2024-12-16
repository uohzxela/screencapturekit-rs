use rodio::{Decoder, OutputStream, Source};
use std::io::{self, Write};
use std::sync::mpsc::{self, Sender, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{BufferSize, SampleRate, Stream, StreamConfig};
use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType};
use hound;
use swap_buffer_queue::{buffer::VecBuffer, Queue};

struct AudioAsyncMic {
    queue: Arc<Mutex<Vec<f32>>>,
    processing_handle: Option<JoinHandle<()>>,
    stream: Option<Stream>,
    tx: Option<SyncSender<Vec<f32>>>,
}

impl AudioAsyncMic {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
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
        let audio_buffer = Arc::new(Mutex::new(Vec::new()));
        let audio_buffer_clone = Arc::clone(&audio_buffer);

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

    fn start(&self) {
        if let Some(ref stream) = self.stream {
            stream.play().unwrap();
        }
    }

    fn stop(&mut self) {
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
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut audio = AudioAsyncMic::new()?;
    audio.start();
    // Wait for user input to terminate.
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();

    audio.stop();

    // Write accumulated audio to WAV file
    let audio_data = audio.queue.lock().unwrap();
    write_wav_file(&audio_data, 16000, 1)?; // 16 kHz, mono

    Ok(())
}

/// Process audio chunk and store it in a shared buffer
fn process_audio(audio_chunk: &[f32], audio_buffer: &Arc<Mutex<Vec<f32>>>) {
    // Acquire the lock and append the chunk to the global buffer
    let mut buffer = audio_buffer.lock().unwrap();
    buffer.extend_from_slice(audio_chunk);
}

/// Write audio samples to a WAV file
fn write_wav_file(samples: &[f32], sample_rate: u32, channels: u16) -> Result<(), hound::Error> {
    // Create WAV writer with the given specifications
    let spec = hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    // Open a WAV file for writing
    let mut writer = hound::WavWriter::create("recorded_audio.wav", spec)?;

    // Write each sample to the WAV file
    for &sample in samples {
        writer.write_sample(sample)?;
    }

    println!("Audio saved to recorded_audio.wav");
    Ok(())
}