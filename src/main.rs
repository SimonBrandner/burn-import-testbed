mod models;

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::NdArray;
//use burn::backend::{wgpu::WgpuDevice, Wgpu};
use burn::tensor::Tensor;
use models::{detector, eye_blink_classifier, recognizer};
use std::time::{Duration, Instant};

const ITERATIONS: u32 = 1;

//type Backend = Wgpu<f32, i32>;
type Backend = NdArray<f32>;

fn main() {
	//let device = WgpuDevice::default();
	let device = NdArrayDevice::default();

	let detector: detector::Model<Backend> = detector::Model::default();
	let recognizer: recognizer::Model<Backend> = recognizer::Model::default();
	let eye_blink_classifier: eye_blink_classifier::Model<Backend> =
		eye_blink_classifier::Model::default();

	let detector_input = Tensor::<Backend, 4>::zeros([1, 3, 480, 640], &device);
	let recognizer_input = Tensor::<Backend, 4>::zeros([1, 3, 128, 128], &device);
	let eye_blink_classifier_input = Tensor::<Backend, 4>::zeros([1, 34, 26, 1], &device);

	let mut detector_sum = Duration::new(0, 0);
	let mut recognizer_sum = Duration::new(0, 0);
	let mut eye_blink_classifier_sum = Duration::new(0, 0);
	for _ in 0..ITERATIONS {
		let mut now = Instant::now();
		detector.forward(detector_input.clone());
		detector_sum += now.elapsed();

		now = Instant::now();
		recognizer.forward(recognizer_input.clone());
		recognizer_sum += now.elapsed();

		now = Instant::now();
		eye_blink_classifier.forward(eye_blink_classifier_input.clone());
		eye_blink_classifier_sum += now.elapsed();
	}

	println!("Detector took on average: {:?}", detector_sum / ITERATIONS);
	println!(
		"Recognizer took on average: {:?}",
		recognizer_sum / ITERATIONS
	);
	println!(
		"Eye-blink classifier took on average: {:?}",
		eye_blink_classifier_sum / ITERATIONS
	);
}
