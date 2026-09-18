# Веса модели

Здесь ожидается `wheat_model.pth`. Фиктивные веса в проект не включены.
Запустите `python scripts/train_model.py --data dataset` или поместите собственные доверенные веса MobileNetV3 Small с четырьмя выходами.

Поддерживаются `state_dict` и checkpoint с `model_state_dict`. Порядок выходов строго:
`healthy`, `leaf_rust`, `septoria`, `yellow_rust` (алфавит ImageFolder).
Checkpoint с полем `classes` проверяется автоматически. У голого state_dict метаданных нет: соответствие порядка обязан обеспечить автор весов.

Используйте те же Resize(256), CenterCrop(224), ToTensor и нормализацию ImageNet, что в `utils/model_config.py`.
Не загружайте файлы из недоверенных источников. Загрузка использует `weights_only=True`.
После замены весов кэш обновляется по размеру и времени изменения файла; при необходимости перезапустите Streamlit.
