# <center> KimchiSpeech: Faster, Lighter and More Controllable TTS with Hierachical VAE and STT-aided Gaussian Attention </center>

<center> Soochul Park </center>

<center> MODULABS, Seoul, Korea </center>

<center> scpark20@gmail.com </center>

![alt text](pics/kimchispeech.png)

## Abstract
In this study, we propose a text-to-speech (TTS) model, referred to as KimchiSpeech, which has a hierarchical variational autoencoder (VAE) structure and uses a attention alignment obtained from a speech-to-text (STT) model. The hierarchical VAE structure contributes to the generation of high-quality outputs and a variety of prosody. Because the STT model operates independently of the TTS model, the attention alignment can be obtained robustly. Moreover, the speed of a generated
speech can be flexibly controlled by soft attention using Gaussian distributions. Furthermore, we propose two configurations of the model, namely KimchiSpeech-W and KimchiSpeech-S. The former is a light version that has only 3.3M parameters for inference, whereas the latter is a fast version that produces outputs 470 times faster than real time on a GPU. The mean opinion score (MOS) results show that the outputs are of state-of-the-art quality.
<audio src="wavs/kimchispeech_abstract.wav" controls preload></audio>

## Text Examples

Text 1: The Middle Ages brought calligraphy to perfection, and it was natural therefore.

| **GT** | **Tacoton2** |
| :--- | :--- |
| <audio src="wavs/GT/inference_0_GT.wav" controls preload></audio> | <audio src="wavs/TACO2/inference_0_TACO2.wav" controls preload></audio> |
| **FastSpeech** | **FastSpeech2** | 
| :--- | :--- |
| <audio src="wavs/FS/inference_0_FS.wav" controls preload></audio> | <audio src="wavs/FS2/inference_0_FS2.wav" controls preload></audio> |
| **KimchiSpeech-W** | **KimchiSpeech-S** |
| :--- | :--- |
| <audio src="wavs/KS/inference_0_S5G1.wav" controls preload></audio> | <audio src="wavs/KW/inference_0_W4G1.wav" controls preload></audio> |
| --- | --- |

Text 2: that the forms of printed letters should follow more or less closely those of the written character, and they followed them very closely.

| **GT** | **Tacoton2** |
| :--- | :--- |
| <audio src="wavs/GT/inference_1_GT.wav" controls preload></audio> | <audio src="wavs/TACO2/inference_1_TACO2.wav" controls preload></audio> |
| **FastSpeech** | **FastSpeech2** | 
| :--- | :--- |
| <audio src="wavs/FS/inference_1_FS.wav" controls preload></audio> | <audio src="wavs/FS2/inference_1_FS2.wav" controls preload></audio> |
| **KimchiSpeech-W** | **KimchiSpeech-S** |
| :--- | :--- |
| <audio src="wavs/KS/inference_1_S5G1.wav" controls preload></audio> | <audio src="wavs/KW/inference_1_W4G1.wav" controls preload></audio> |
| --- | --- |

Text 3: especially as regards the lower case letters; and type very similar was used during the next fifteen or twenty years not only by Schoeffer,

| **GT** | **Tacoton2** |
| :--- | :--- |
| <audio src="wavs/GT/inference_2_GT.wav" controls preload></audio> | <audio src="wavs/TACO2/inference_2_TACO2.wav" controls preload></audio> |
| **FastSpeech** | **FastSpeech2** | 
| :--- | :--- |
| <audio src="wavs/FS/inference_2_FS.wav" controls preload></audio> | <audio src="wavs/FS2/inference_2_FS2.wav" controls preload></audio> |
| **KimchiSpeech-W** | **KimchiSpeech-S** |
| :--- | :--- |
| <audio src="wavs/KS/inference_2_S5G1.wav" controls preload></audio> | <audio src="wavs/KW/inference_2_W4G1.wav" controls preload></audio> |
| --- | --- |

Text 4: a very few years saw the birth of Roman character not only in Italy, but in Germany and France.

| **GT** | **Tacoton2** |
| :--- | :--- |
| <audio src="wavs/GT/inference_3_GT.wav" controls preload></audio> | <audio src="wavs/TACO2/inference_3_TACO2.wav" controls preload></audio> |
| **FastSpeech** | **FastSpeech2** | 
| :--- | :--- |
| <audio src="wavs/FS/inference_3_FS.wav" controls preload></audio> | <audio src="wavs/FS2/inference_3_FS2.wav" controls preload></audio> |
| **KimchiSpeech-W** | **KimchiSpeech-S** |
| :--- | :--- |
| <audio src="wavs/KS/inference_3_S5G1.wav" controls preload></audio> | <audio src="wavs/KW/inference_3_W4G1.wav" controls preload></audio> |
| --- | --- |

Text 5: and used an exceedingly beautiful type, which is indeed to look at a transition between Gothic and Roman,

| **GT** | **Tacoton2** |
| :--- | :--- |
| <audio src="wavs/GT/inference_4_GT.wav" controls preload></audio> | <audio src="wavs/TACO2/inference_4_TACO2.wav" controls preload></audio> |
| **FastSpeech** | **FastSpeech2** | 
| :--- | :--- |
| <audio src="wavs/FS/inference_4_FS.wav" controls preload></audio> | <audio src="wavs/FS2/inference_4_FS2.wav" controls preload></audio> |
| **KimchiSpeech-W** | **KimchiSpeech-S** |
| :--- | :--- |
| <audio src="wavs/KS/inference_4_S5G1.wav" controls preload></audio> | <audio src="wavs/KW/inference_4_W4G1.wav" controls preload></audio> |
| --- | --- |

Text 6: John of Spires and his brother Vindelin, followed by Nicholas Jenson, began to print in that city,

| **GT** | **Tacoton2** |
| :--- | :--- |
| <audio src="wavs/GT/inference_5_GT.wav" controls preload></audio> | <audio src="wavs/TACO2/inference_5_TACO2.wav" controls preload></audio> |
| **FastSpeech** | **FastSpeech2** | 
| :--- | :--- |
| <audio src="wavs/FS/inference_5_FS.wav" controls preload></audio> | <audio src="wavs/FS2/inference_5_FS2.wav" controls preload></audio> |
| **KimchiSpeech-W** | **KimchiSpeech-S** |
| :--- | :--- |
| <audio src="wavs/KS/inference_5_S5G1.wav" controls preload></audio> | <audio src="wavs/KW/inference_5_W4G1.wav" controls preload></audio> |
| --- | --- |

Text 7: fourteen sixty nine, fourteen seventy;

| **GT** | **Tacoton2** |
| :--- | :--- |
| <audio src="wavs/GT/inference_6_GT.wav" controls preload></audio> | <audio src="wavs/TACO2/inference_6_TACO2.wav" controls preload></audio> |
| **FastSpeech** | **FastSpeech2** | 
| :--- | :--- |
| <audio src="wavs/FS/inference_6_FS.wav" controls preload></audio> | <audio src="wavs/FS2/inference_6_FS2.wav" controls preload></audio> |
| **KimchiSpeech-W** | **KimchiSpeech-S** |
| :--- | :--- |
| <audio src="wavs/KS/inference_6_S5G1.wav" controls preload></audio> | <audio src="wavs/KW/inference_6_W4G1.wav" controls preload></audio> |
| --- | --- |

Text 8: and though the famous family of Aldus restored its technical excellence, rejecting battered letters,

| **GT** | **Tacoton2** |
| :--- | :--- |
| <audio src="wavs/GT/inference_7_GT.wav" controls preload></audio> | <audio src="wavs/TACO2/inference_7_TACO2.wav" controls preload></audio> |
| **FastSpeech** | **FastSpeech2** | 
| :--- | :--- |
| <audio src="wavs/FS/inference_7_FS.wav" controls preload></audio> | <audio src="wavs/FS2/inference_7_FS2.wav" controls preload></audio> |
| **KimchiSpeech-W** | **KimchiSpeech-S** |
| :--- | :--- |
| <audio src="wavs/KS/inference_7_S5G1.wav" controls preload></audio> | <audio src="wavs/KW/inference_7_W4G1.wav" controls preload></audio> |
| --- | --- |

Text 9: yet their type is artistically on a much lower level than Jenson's, and in fact.

| **GT** | **Tacoton2** |
| :--- | :--- |
| <audio src="wavs/GT/inference_8_GT.wav" controls preload></audio> | <audio src="wavs/TACO2/inference_8_TACO2.wav" controls preload></audio> |
| **FastSpeech** | **FastSpeech2** | 
| :--- | :--- |
| <audio src="wavs/FS/inference_8_FS.wav" controls preload></audio> | <audio src="wavs/FS2/inference_8_FS2.wav" controls preload></audio> |
| **KimchiSpeech-W** | **KimchiSpeech-S** |
| :--- | :--- |
| <audio src="wavs/KS/inference_8_S5G1.wav" controls preload></audio> | <audio src="wavs/KW/inference_8_W4G1.wav" controls preload></audio> |
| --- | --- |

Text 10: they must be considered to have ended the age of fine printing in Italy.

| **GT** | **Tacoton2** |
| :--- | :--- |
| <audio src="wavs/GT/inference_9_GT.wav" controls preload></audio> | <audio src="wavs/TACO2/inference_9_TACO2.wav" controls preload></audio> |
| **FastSpeech** | **FastSpeech2** | 
| :--- | :--- |
| <audio src="wavs/FS/inference_9_FS.wav" controls preload></audio> | <audio src="wavs/FS2/inference_9_FS2.wav" controls preload></audio> |
| **KimchiSpeech-W** | **KimchiSpeech-S** |
| :--- | :--- |
| <audio src="wavs/KS/inference_9_S5G1.wav" controls preload></audio> | <audio src="wavs/KW/inference_9_W4G1.wav" controls preload></audio> |
| --- | --- |

## Speed Control

Text: The quick brown fox jumps over the lazy dog.

| **0.7x** | **0.8x** |
| :--- | :--- |
| <audio src="wavs/fox0.7.wav" controls preload></audio> | <audio src="wavs/fox0.8.wav" controls preload></audio> |
| **0.9x** | **1.0x** | 
| :--- | :--- |
| <audio src="wavs/fox0.9.wav" controls preload></audio> | <audio src="wavs/fox1.0.wav" controls preload></audio> |
| **1.1x** | **1.2x** |
| :--- | :--- |
| <audio src="wavs/fox1.1.wav" controls preload></audio> | <audio src="wavs/fox1.2.wav" controls preload></audio> |
| **1.3x** | **1.4x** |
| :--- | :--- |
| <audio src="wavs/fox1.3.wav" controls preload></audio> | <audio src="wavs/fox1.4.wav" controls preload></audio> |
| --- | --- |

## Truncated Normal Distribution

Text: In probability and statistics, the truncated normal distribution is the probability distribution derived from that of a normally distributed random variable by bounding the random variable from either below or above. (from https://en.wikipedia.org/wiki/Truncated_normal_distribution)

| **range (-3, 3)** | **range (-2, 2)** |
| :--- | :--- |
| <audio src="wavs/tn3.wav" controls preload></audio> | <audio src="wavs/tn2.wav" controls preload></audio> |
| **range (-1, 1)** | **range (-0.01, 0.01)** | 
| :--- | :--- |
| <audio src="wavs/tn1.wav" controls preload></audio> | <audio src="wavs/tn0.01.wav" controls preload></audio> |
| --- | --- |



