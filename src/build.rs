use burn_import::onnx::ModelGen;

const MODEL_NAMES: [&str; 2] = ["detector", "recognizer"];
const MODELS_OUT_DIR: &str = "src/models";
const ONNX_DIR: &str = "models";

fn main() {
	for model_name in MODEL_NAMES {
		ModelGen::new()
			.input(&(String::new() + ONNX_DIR + "/" + model_name + ".onnx"))
			.out_dir(MODELS_OUT_DIR)
			.run_from_script();
	}
}
