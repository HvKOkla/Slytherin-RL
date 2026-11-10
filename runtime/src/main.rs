use ort::session::Session;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Slytherin-RL Rust Inference Engine ---");

    // 1. Charger le modèle ONNX (API simplifiée de rc.10)
    let mut session = Session::builder()?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
        .with_intra_threads(1)?
        .commit_from_file("slytherin_v1_1.onnx")?;

    // 2. Créer une fausse entrée (13 features)
    let input_data = vec![0.1f32; 13]; 
    let input_tensor = ort::value::Tensor::from_array((vec![1, 13], input_data.clone()))?;

    // 3. Warmup
    for _ in 0..100 {
        let _ = session.run(ort::inputs!["input" => &input_tensor])?;
    }

    // 4. Benchmark
    let n_steps = 1000;
    let mut total_time = 0.0;

    for _ in 0..n_steps {
        let start = Instant::now();
        let _outputs = session.run(ort::inputs!["input" => &input_tensor])?;
        total_time += start.elapsed().as_secs_f64() * 1000.0; 
    }

    let avg_latency = total_time / (n_steps as f64);
    println!("✅ Inférence terminée !");
    println!("🚀 Latence moyenne : {:.4} ms", avg_latency);
    println!("📈 Throughput : {:.0} inférences / sec", 1000.0 / avg_latency);

    Ok(())
}