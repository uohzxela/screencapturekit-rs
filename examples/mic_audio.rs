use rodio::{Decoder, OutputStream, Source};
use std::io::{self, Write};
use std::sync::mpsc::{self, Sender};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleRate, StreamConfig, BufferSize};
use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
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

    stream.play()?;
    println!("Audio stream started. Processing audio...\nPress Enter to stop.");

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
                    process_audio(&resampled_chunk);
                }
            } else {
                eprintln!("Error during resampling.");
            }
        }
    });

    // Wait for user input to terminate.
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();

    println!("Stopping audio stream...");
    drop(stream); // Dropping the stream stops it.

    // Ensure the processing thread finishes.
    drop(tx); // Closing the channel.
    processing_handle.join().unwrap();

    Ok(())
}

/// Dummy audio processing function.
/// Replace this with your actual audio processing logic.
fn process_audio(audio_chunk: &[f32]) {
    println!("Processing resampled audio chunk of size: {}", audio_chunk.len());

    // Example: Write audio data to a file or analyze it.
    // For simplicity, just print the first few samples.
    for sample in audio_chunk.iter().take(10) {
        print!("{} ", sample);
    }
    println!();
}
