"""
Основной файл с кодом бота.
Состоит из функций-обработчиков с декораторами (фильтрами).
"""

from aiogram import F, Router, Bot
from aiogram.filters import Command
from aiogram.types import Message, CallbackQuery

from aiogram import flags
from aiogram.fsm.context import FSMContext
import io
import librosa

from states import Intonation
from utils import EmotionRecognition, check_voiced
import kb
import text
import random

router = Router()
emotion_recognizer = EmotionRecognition()


@router.message(Command("start"))
async def start_handler(msg: Message, state: FSMContext):
    """Обработчик команды \\start"""
    await state.set_state(Intonation.menu)
    await msg.answer(text.greet, reply_markup=kb.menu)


# MENU

@router.message(F.text == "Меню")
@router.message(F.text == "меню")
@router.message(F.text == "◀️ Выйти в меню")
async def menu(msg: Message, state: FSMContext):
    """Обработчик текстовой команды 'Меню'"""
    await state.set_state(Intonation.menu)
    await msg.answer(text.menu, reply_markup=kb.menu)


# HELP

@router.callback_query(F.data == "help")
async def menu_help_handler(clbck: CallbackQuery):
    """Обработчик команды 'Помощь' (кнопка интерактивной клавиатуры)"""
    await clbck.message.edit_text(text.help_instruction, reply_markup=kb.menu)


@router.message(Command("help"))
@router.message(Intonation.menu, F.voice)
async def help_handler(msg: Message):
    """Обработчик команды \\help или голосового сообщения вне тренировки"""
    await msg.answer(text.help_instruction, reply_markup=kb.menu)


# HAPPY

@router.callback_query(F.data == "happy")
async def input_happy(clbck: CallbackQuery, state: FSMContext):
    """Переход в режим тренировки эмоции 'Радость'"""
    await state.set_state(Intonation.happy)
    await clbck.message.answer(text.happy_instruction, reply_markup=kb.exit_kb)
    await clbck.message.answer(text.exit_instruction, reply_markup=kb.start_kb)


@router.callback_query(Intonation.happy, F.data == "next_phrase")
async def next_phrase_happy(clbck: CallbackQuery):
    """Выдача следующей случайной фразы из text.phrases в режиме 'Радость'"""
    await clbck.message.edit_text("😄: {}".format(text.phrases[random.randint(0, len(text.phrases))]))


@router.message(Intonation.happy, F.voice)
@flags.chat_action("typing")
async def voice_file_happy(msg: Message, bot: Bot):
    """Обработка голосового сообщения в режиме 'Радость': \n
    1. Проверка наличия тоновых звуков в записи, \n
    2. Определение вероятностей для каждой эмоции, \n
    3. Выдача результата пользователю"""
    file_id = msg.voice.file_id
    file = await bot.get_file(file_id)
    await msg.answer(text.wait_feedback)
    audio: io.BytesIO = await bot.download_file(file.file_path)
    y, sr = librosa.load(audio, sr=16000)
    if check_voiced(y, sr):
        probs = emotion_recognizer.recognize(y, sr)
        await msg.answer(text.result.format(*probs))
        await msg.answer(text.ask_continue, reply_markup=kb.continue_kb)
    else:
        await msg.answer(text.unvoiced)


@router.message(Intonation.happy)
async def other_happy(msg: Message):
    """Обработчик прочих сообщений в режиме 'Радость'. \n
    Выдает инструкцию по тренировке."""
    await msg.answer(text.happy_instruction, reply_markup=kb.continue_kb)


# SAD

@router.callback_query(F.data == "sad")
async def input_sad(clbck: CallbackQuery, state: FSMContext):
    """Переход в режим тренировки эмоции 'Грусть'"""
    await state.set_state(Intonation.sad)
    await clbck.message.answer(text.sad_instruction)
    await clbck.message.answer(text.exit_instruction, reply_markup=kb.start_kb)


@router.callback_query(Intonation.sad, F.data == "next_phrase")
async def next_phrase_sad(clbck: CallbackQuery):
    """Выдача следующей случайной фразы из модуля text в режиме 'Грусть'"""
    await clbck.message.edit_text("😔: {}".format(text.phrases[random.randint(0, len(text.phrases))]))


@router.message(Intonation.sad, F.voice)
@flags.chat_action("typing")
async def voice_file_sad(msg: Message, bot: Bot):
    """Обработка голосового сообщения в режиме 'Грусть': \n
        1. Проверка наличия тоновых звуков в записи, \n
        2. Определение вероятностей для каждой эмоции, \n
        3. Выдача результата пользователю"""
    file_id = msg.voice.file_id
    file = await bot.get_file(file_id)
    await msg.answer(text.wait_feedback)
    audio: io.BytesIO = await bot.download_file(file.file_path)
    y, sr = librosa.load(audio, sr=16000)
    probs = emotion_recognizer.recognize(y, sr)

    await msg.answer(text.result.format(*probs))
    await msg.answer(text.ask_continue, reply_markup=kb.continue_kb)


@router.message(Intonation.sad)
async def other_sad(msg: Message):
    """Обработчик прочих сообщений в режиме 'Грусть'. \n
        Выдает инструкцию по тренировке."""
    await msg.answer(text.sad_instruction, reply_markup=kb.continue_kb)


# ANGRY

@router.callback_query(F.data == "angry")
async def input_angry(clbck: CallbackQuery, state: FSMContext):
    """Переход в режим тренировки эмоции 'Злость'"""
    await state.set_state(Intonation.angry)
    await clbck.message.answer(text.angry_instruction)
    await clbck.message.answer(text.exit_instruction, reply_markup=kb.start_kb)


@router.callback_query(Intonation.angry, F.data == "next_phrase")
async def next_phrase_angry(clbck: CallbackQuery):
    """Выдача следующей случайной фразы из модуля text в режиме 'Злость'"""
    await clbck.message.edit_text("😠: {}".format(text.phrases[random.randint(0, len(text.phrases))]))


@router.message(Intonation.angry, F.voice)
@flags.chat_action("typing")
async def voice_file_angry(msg: Message, bot: Bot):
    """Обработка голосового сообщения в режиме 'Злость': \n
        1. Проверка наличия тоновых звуков в записи, \n
        2. Определение вероятностей для каждой эмоции, \n
        3. Выдача результата пользователю"""
    file_id = msg.voice.file_id
    file = await bot.get_file(file_id)
    await msg.answer(text.wait_feedback)
    audio: io.BytesIO = await bot.download_file(file.file_path)
    y, sr = librosa.load(audio, sr=16000)
    probs = emotion_recognizer.recognize(y, sr)

    await msg.answer(text.result.format(*probs))
    await msg.answer(text.ask_continue, reply_markup=kb.continue_kb)


@router.message(Intonation.angry)
async def other_angry(msg: Message):
    """Обработчик прочих сообщений в режиме 'Злость'. \n
        Выдает инструкцию по тренировке."""
    await msg.answer(text.angry_instruction, reply_markup=kb.continue_kb)
