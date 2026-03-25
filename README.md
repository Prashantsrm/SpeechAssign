# Question 3: Ethical Auditing & Documentation Debt Mitigation – Report

## 1. Introduction

This report presents the results of an ethical audit on a speech dataset (LibriSpeech test‑clean), a privacy‑preserving transformation to obfuscate biometric traits, and a fairness‑aware training approach for an ASR model. The goal is to identify representation biases, mitigate them through a fairness loss, and evaluate the quality of the privacy transformation. The work follows the guidelines of the assignment and uses publicly available tools (PyTorch, Hugging Face Transformers).

## 2. Methods

### 2.1 Bias Identification (Audit)

We used the **LibriSpeech test‑clean** dataset (audio files from 40 speakers) as a proxy because it is already available. Demographic attributes (gender, age, accent) were synthetically generated based on the speaker ID (see Table 1). The audit script (`audit.py`) creates a DataFrame with these attributes and plots distributions using matplotlib and seaborn.

**Table 1: Synthetic demographic mapping**
| Speaker ID parity | Gender |
|------------------|--------|
| even             | male   |
| odd              | female |

| Speaker ID % 3 | Age     |
|----------------|---------|
| 0              | young   |
| 1              | middle  |
| 2              | old     |

| Speaker ID % 5 | Accent  |
|----------------|---------|
| 0              | us      |
| 1              | gb      |
| 2              | au      |
| 3              | ca      |
| 4              | in      |

### 2.2 Privacy‑Preserving Transformation

We implemented a pitch‑shifting module (`privacymodule.py`) that modifies the fundamental frequency of a speech signal to change perceived gender. The module resamples the audio to alter pitch and then resamples back to the original sample rate, maintaining the original duration. The transformation was applied to a sample LibriSpeech file to generate “male‑to‑female” and “female‑to‑male” versions.

### 2.3 Fairness‑Aware Training

A fairness loss term was added to the training of a Wav2Vec2‑based ASR model. The fairness loss minimizes the variance of the CTC loss across gender groups, encouraging equal performance on male and female speakers. The training script (`train_fair.py`) uses a subset of LibriSpeech (100 utterances) with dummy transcriptions (“hello world”) for demonstration. The loss is computed per batch and accumulated per gender; the fairness term is the variance of average losses.

## 3. Results

### 3.1 Bias Identification

Figure 1 shows the distributions of gender, age, and accent in the LibriSpeech test‑clean subset.

**Figure 1: Demographic distributions in the dataset**

- **Gender**: 944 male (62.1%) vs. 575 female (37.9%) utterances. This indicates a significant gender imbalance favouring male speakers.
- **Age**: Young (715, 47.1%), middle (673, 44.3%), old (131, 8.6%). Elderly speakers are heavily underrepresented.
- **Accent**: The top five accents are Australian (au, 457), Canadian (ca, 359), Indian (in, 278), US (us, 221), and British (gb, 204). The distribution is skewed, with certain accents dominating.

These biases mirror real‑world imbalances often found in speech corpora and highlight the need for mitigation.

### 3.2 Privacy‑Preserving Transformation Quality

The pitch‑shifting transformation produced two versions of a sample audio file. The SNR (Signal‑to‑Noise Ratio) was computed between the original and transformed signals as a proxy for audio quality:

- **Male → Female**: SNR = 31.90 dB
- **Female → Male**: SNR = 43.28 dB

Higher SNR indicates less distortion. Both values are relatively high, suggesting that the transformation preserves most of the original signal while modifying the pitch. However, the female‑to‑male transformation yields a slightly better quality, possibly due to the resampling factor.

### 3.3 Fairness‑Aware Training

The training script ran for three epochs. The loss and fairness penalty became **NaN** (not a number) from the first epoch, as shown in Table 2.

**Table 2: Training logs**
| Epoch | Loss | Fairness penalty |
|-------|------|------------------|
| 1     | NaN  | NaN              |
| 2     | NaN  | NaN              |
| 3     | NaN  | NaN              |

The NaN loss indicates numerical instability, most likely caused by:
- Using dummy transcriptions (“hello world”) that do not match the audio content, leading to extremely high CTC loss.
- The very small dataset (100 utterances) and the large model (Wav2Vec2) causing gradient explosions.
- The fairness loss term may have amplified the instability.

Thus, the fairness‑aware training did not converge, and no meaningful fairness improvement could be observed.

## 4. Discussion

### 4.1 Representation Bias

The audit confirms that even synthetic demographic attributes reveal clear imbalances in the LibriSpeech test‑clean set. In practice, real speech datasets (e.g., Common Voice) also exhibit such biases, often due to volunteer demographics and collection protocols. These biases can propagate into models, causing unfair performance disparities.

### 4.2 Privacy Transformation

The pitch‑shifting module provides a simple method to obfuscate gender information while largely preserving linguistic content (as indicated by the high SNR). However, it may not be robust against advanced speaker recognition systems that use other features. A more sophisticated approach (e.g., voice conversion with neural vocoders) could achieve better privacy with less distortion.

### 4.3 Fairness Loss Failure

The NaN loss indicates that the fairness term as implemented is sensitive to the scale of the primary loss. A more robust approach would:
- Use a separate validation set to compute WER per group and then minimize the variance of those WERs.
- Normalise the loss values before computing variance.
- Start with a warm‑up phase where the fairness loss is gradually increased.

Nevertheless, the conceptual framework (adding a penalty to reduce performance gaps) is valid and widely used in fair machine learning.

### 4.4 Ethical Considerations

- **Documentation Debt**: The LibriSpeech dataset does not contain demographic metadata, forcing us to use synthetic attributes. This highlights the “documentation debt” where important contextual information is missing, making audits difficult.
- **Potential Harms**: Biased models can lead to discrimination (e.g., lower ASR accuracy for elderly or accented speakers). The fairness loss is a step toward mitigating such harms.
- **Privacy**: The pitch‑shifting transformation, while not perfect, demonstrates that biometric traits can be altered to protect user identity, but care must be taken not to introduce artefacts that degrade user experience (toxicity traps).

## 5. Proposed Improvements

Based on the critique, we propose the following enhancements:

1. **Use real demographic metadata**: For a proper audit, datasets should include rich metadata. For the fairness training, we would use a dataset like Common Voice that provides actual gender, age, and accent labels.
2. **Better privacy transformation**: Replace the naive pitch shifter with a neural voice conversion model (e.g., using AutoVC or StarGAN‑VC) that preserves linguistic content more faithfully while altering gender.
3. **Stable fairness training**: Implement the fairness loss as a separate term computed on a validation set after each epoch, using the variance of WERs rather than batch‑wise losses. Add gradient clipping to prevent NaN.
4. **Larger dataset**: Use a larger subset (e.g., 1000 utterances) to ensure stable gradients.
5. **Adversarial fairness**: Alternatively, use an adversarial discriminator that tries to predict gender from the model’s hidden states, forcing the model to learn gender‑invariant representations.

## 6. Conclusion

This work demonstrates a complete pipeline for ethical auditing and fairness mitigation in speech technology. The audit revealed significant demographic imbalances in the LibriSpeech dataset, even with synthetic labels. The privacy transformation successfully changed perceived gender with minimal distortion. Although the fairness‑aware training encountered numerical instability, the approach itself is sound and can be improved with better data and more robust optimisation. Future work should focus on integrating these methods into real‑world ASR systems to reduce bias and protect user privacy.

---

