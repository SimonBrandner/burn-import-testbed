mod models;

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::NdArray;
use burn::backend::{wgpu::WgpuDevice, Wgpu};
use burn::tensor::Tensor;
use models::{detector, recognizer};
use std::time::{Duration, Instant};

const ITERATIONS: u32 = 64;

//type Backend = Wgpu<f32, i32>;
type Backend = NdArray<f32>;

fn main() {
	//let device = WgpuDevice::default();
	let device = NdArrayDevice::default();

	let detector: detector::Model<Backend> = detector::Model::new(&device);
	let recognizer: recognizer::Model<Backend> = recognizer::Model::new(&device);

	let detector_input = Tensor::<Backend, 4>::zeros([1, 3, 480, 640], &device);
	let recognizer_input = Tensor::<Backend, 4>::zeros([1, 3, 128, 128], &device);

	let mut detector_sum = Duration::new(0, 0);
	let mut recognizer_sum = Duration::new(0, 0);
	for _ in 0..ITERATIONS {
		let mut now = Instant::now();
		detector.forward(detector_input.clone());
		detector_sum += now.elapsed();

		now = Instant::now();
		recognizer.forward(recognizer_input.clone());
		recognizer_sum += now.elapsed();
	}

	println!("Detector took on average: {:?}", detector_sum / ITERATIONS);
	println!(
		"Recognizer took on average: {:?}",
		recognizer_sum / ITERATIONS
	);
}
