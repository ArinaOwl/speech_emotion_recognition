"""
Вспомогательные функции
"""

import numpy as np
import torch
from transformers import AutoFeatureExtractor, ASTForAudioClassification
import librosa


class EmotionRecognition:
    """Класс инференса модели."""
    def __init__(self):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("ArinaOwl/ast-ser-ru")
        self.model = ASTForAudioClassification.from_pretrained("ArinaOwl/ast-ser-ru")
        self.model.eval()
        self.to_probs = torch.nn.Softmax(dim=0).requires_grad_(False)

    def recognize(self, waveform, sampling_rate):
        """Инференс.

        Параметры:
            - waveform (np.ndarray): звуковой временной ряд,
            - sampling_rate (int): частота дискретизации.

        Возвращает:
            probs (np.ndarray) - вероятности распознавания для каждой эмоции."""
        features = self.feature_extractor(waveform, sampling_rate, return_tensors='pt')
        outputs, = self.model(features.input_values.to(torch.float32))
        probs = self.to_probs(outputs[0])
        return np.round(probs.detach().numpy() * 100).astype(int)


def check_voiced(waveform, sampling_rate):
    """Проверка наличия тоновых звуков в записи.

    Параметры:
        - waveform (np.ndarray): звуковой временной ряд,
        - sampling_rate (int): частота дискретизации.

    Возвращает:
        True, если обнаружен голос, иначе False.
    """
    f0, voicing, voicing_p = librosa.pyin(waveform, sr=sampling_rate,
                                          fmin=64., fmax=1046.,
                                          frame_length=512, fill_na=0.)
    return np.max(f0) > 0
