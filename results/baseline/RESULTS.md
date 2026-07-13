# Baseline Experiment — Results

End-to-end baseline run of the qualia architecture on the synthetic
``ColoredShapesDataset`` (encoder + predictive loop + self-model + decoder).

## Training losses

| Metric                 |   step 0 |   final |
|------------------------|---------:|--------:|
| train loss             | 0.7555 | 0.6754 |
| reconstruction (pred)  | 0.0143 | 0.0010 |
| decoder reconstruction | 0.6319 | 0.5718 |
| predictive-coding KL   | 0.0074 | 0.0125 |
| report consistency     | 0.0000 | 0.0000 |
| introspection loss     | 1.0853 | 1.0139 |

## Eval suite

| Metric                 |   value |
|------------------------|--------:|
| report consistency     | 0.2167 |
| introspection accuracy | 0.2083 |
| downstream grounding   | 0.0000 |

## Run summary

- **steps**: `100`
- **batch_size**: `16`
- **lr**: `0.0003`
- **train_elapsed_s**: `2.98`
- **eval_elapsed_s**: `0.32`
- **final_train_loss**: `0.6754`
- **introspection_accuracy**: `0.2083`
- **report_consistency**: `0.2167`
- **downstream_grounding**: `0.0000`
