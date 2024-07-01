use std::env;
use std::fs::{self, File};
use std::io::{self, Write};
// use std::path::Path;

fn main() -> io::Result<()> {
    // Retrieve the path to the binary file from the environment variable
    let model_path = env::var("AI_MODEL").expect("AI_MODEL environment variable not set");

    // Read the binary file contents
    let binary_content = fs::read(&model_path)?;
    
    // Start writing to `model.rs`
    let mut output_file = File::create("src/ai_model.rs")?;

    // Write the static array declaration to `model.rs`
    write!(
        &mut output_file,
        "pub static MODEL: &'static [u8] = &[\n"
    )?;

    // Write the binary contents as a byte array
    for chunk in binary_content.chunks(16) {
        let byte_strings: Vec<String> = chunk.iter().map(|b| format!("{:#04X}", b)).collect();
        writeln!(&mut output_file, "    {},", byte_strings.join(", "))?;
    }

    // Close the array and the static declaration
    write!(&mut output_file, "];\n")?;

    Ok(())
}