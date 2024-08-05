use burn_import::onnx::ModelGen;

const MODELS_OUTPUT_DIRECTORY: &str = "src/models";
const MODELS_INPUT_DIRECTORY: &str = "models";

fn get_onnx_file_path(model_name: &str) -> String {
	format!("{}/{}.onnx", MODELS_INPUT_DIRECTORY, model_name)
}

fn main() {
	println!("cargo::rerun-if-changed=models");
	ModelGen::new()
		//.input(&get_onnx_file_path("depth_classifier"))
		//.input(&get_onnx_file_path("landmarks_detector"))
		//.input(&get_onnx_file_path("yolov5s-detector"))
		.input(&get_onnx_file_path("detector"))
		.input(&get_onnx_file_path("eye_blink_classifier"))
		.input(&get_onnx_file_path("recognizer"))
		.out_dir(MODELS_OUTPUT_DIRECTORY)
		.run_from_script();
}
