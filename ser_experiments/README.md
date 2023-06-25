# Результаты экспериментов

## Анализ и предобработка данных

### Основная идея

**Частота основного тона** (ЧОТ), Fundamental Frequency или F0 – это частота колебания голосовых связок. При произнесении речи она непрерывно меняется в соответствии с ударением, подчеркиванием звуков и слов, а также при проявлении эмоций. Изменение частоты основного тона называют интонацией. 

Изменение ЧОТ можно увидеть на спектрограмме. Поэтому для экспериментов была выбрана архитектура [Audio Spectrogram Transformer (AST)](https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer).

<img src="https://github.com/ArinaOwl/speech_emotion_recognition/blob/master/ser_experiments/images/spectrogram.png" alt="drawing" width="800"/>

### Агрегация набора данных Dusha

Для обучения и тестирования моделей нейросетевого распознавания эмоций в аудиозаписях был выбран открытый набор данных [Dusha](https://github.com/salute-developers/golos/tree/master/dusha).

В нем данные разделены на две категории:

- Crowd - собраны с помощью краудсорсинга;
- Podcast - короткие нарезки русскоязычных подкастов.

Выборка Crowd подходит под задачу бота, определяющего эмоции в голосовых сообщениях, так как дикторы специально записывали подготовленные фразы.

Данные в категории Crowd разделены на тренировочную и тестовую выборки так, чтобы одинаковые дикторы или подкасты не попали в обе выборки.

Алгоритм агрегации данных подробно описан в [ноутбуке](https://github.com/ArinaOwl/speech_emotion_recognition/blob/master/ser_experiments/data_processing.ipynb). В итоговую выборку вошли аудиозаписи, для которых:
- не менее 5 разметчиков,
- длительность не более 10 секунд,
- эмоция определяется большинством голосов разметчиков,
- определяется ЧОТ.

### Выделение признаков

Для проведения экспериментов были выделены и сохранены признаки аудиозаписей из агрегированного набора:
- для AST

<img src="https://github.com/ArinaOwl/speech_emotion_recognition/blob/master/ser_experiments/images/ast_features.png" alt="drawing" width="800"/>

- для своей архитектуры

<img src="https://github.com/ArinaOwl/speech_emotion_recognition/blob/master/ser_experiments/images/f0_features.png" alt="drawing" width="800"/>

## Отчет о проведенных экспериментах

Проведенные эксперименты подробно описаны в [ноутбуке](https://github.com/ArinaOwl/speech_emotion_recognition/blob/master/ser_experiments/emo_classification_pretrained.ipynb).

Итоговые показатели выбранной при проведении экспериментов модели:

|	          | macro_average |	positive |	sad   |	angry |	neutral |
|:---       | ---:          | ---:     | ---:   | ---:  | ---:    |
| accuracy  |	0.669         |	0.637    |	0.671 |	0.686 |	0.683   |
| precision |	0.679         |	0.615    |	0.700 |	0.736 |	0.667   |
| recall    |	0.677         |	0.721    |	0.730 |	0.729 |	0.529   |
| f1        |	0.675         |	0.664    |	0.715 |	0.733 |	0.590   |
