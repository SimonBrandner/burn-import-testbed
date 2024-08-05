pub mod detector {
	include!(concat!(env!("OUT_DIR"), "/src/models/detector.rs"));
}

pub mod recognizer {
	include!(concat!(env!("OUT_DIR"), "/src/models/recognizer.rs"));
}

pub mod eye_blink_classifier {
	include!(concat!(
		env!("OUT_DIR"),
		"/src/models/eye_blink_classifier.rs"
	));
}
