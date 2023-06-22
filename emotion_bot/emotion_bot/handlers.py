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
    await state.set_state(Intonation.menu)
    await msg.answer(text.greet, reply_markup=kb.menu)


# MENU

@router.message(F.text == "Меню")
@router.message(F.text == "меню")
@router.message(F.text == "◀️ Выйти в меню")
async def menu(msg: Message, state: FSMContext):
    await state.set_state(Intonation.menu)
    await msg.answer(text.menu, reply_markup=kb.menu)


# HELP

@router.callback_query(F.data == "help")
async def menu_help_handler(clbck: CallbackQuery):
    await clbck.message.edit_text(text.help_instruction, reply_markup=kb.menu)


@router.message(Command("help"))
@router.message(Intonation.menu, F.voice)
async def help_handler(msg: Message):
    await msg.answer(text.help_instruction, reply_markup=kb.menu)


# HAPPY

@router.callback_query(F.data == "happy")
async def input_happy(clbck: CallbackQuery, state: FSMContext):
    await state.set_state(Intonation.happy)
    await clbck.message.answer(text.happy_instruction, reply_markup=kb.exit_kb)
    await clbck.message.answer(text.exit_instruction, reply_markup=kb.start_kb)


@router.callback_query(Intonation.happy, F.data == "next_phrase")
async def next_phrase_happy(clbck: CallbackQuery):
    await clbck.message.edit_text("😄: {}".format(text.phrases[random.randint(0, len(text.phrases))]))


@router.message(Intonation.happy, F.voice)
@flags.chat_action("typing")
async def happy_voice_file(msg: Message, bot: Bot):
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
async def happy_other(msg: Message):
    await msg.answer(text.happy_instruction, reply_markup=kb.continue_kb)


# SAD

@router.callback_query(F.data == "sad")
async def input_sad(clbck: CallbackQuery, state: FSMContext):
    await state.set_state(Intonation.sad)
    await clbck.message.answer(text.sad_instruction)
    await clbck.message.answer(text.exit_instruction, reply_markup=kb.start_kb)


@router.callback_query(Intonation.sad, F.data == "next_phrase")
async def next_phrase_sad(clbck: CallbackQuery):
    await clbck.message.edit_text("😔: {}".format(text.phrases[random.randint(0, len(text.phrases))]))


@router.message(Intonation.sad, F.voice)
@flags.chat_action("typing")
async def sad_voice_file(msg: Message, bot: Bot):
    file_id = msg.voice.file_id
    file = await bot.get_file(file_id)
    await msg.answer(text.wait_feedback)
    audio: io.BytesIO = await bot.download_file(file.file_path)
    y, sr = librosa.load(audio, sr=16000)
    probs = emotion_recognizer.recognize(y, sr)

    await msg.answer(text.result.format(*probs))
    await msg.answer(text.ask_continue, reply_markup=kb.continue_kb)


@router.message(Intonation.sad)
async def sad_other(msg: Message):
    await msg.answer(text.sad_instruction, reply_markup=kb.continue_kb)


# ANGRY

@router.callback_query(F.data == "angry")
async def input_angry(clbck: CallbackQuery, state: FSMContext):
    await state.set_state(Intonation.angry)
    await clbck.message.answer(text.angry_instruction)
    await clbck.message.answer(text.exit_instruction, reply_markup=kb.start_kb)


@router.callback_query(Intonation.angry, F.data == "next_phrase")
async def next_phrase_angry(clbck: CallbackQuery):
    await clbck.message.edit_text("😠: {}".format(text.phrases[random.randint(0, len(text.phrases))]))


@router.message(Intonation.angry, F.voice)
@flags.chat_action("typing")
async def angry_voice_file(msg: Message, bot: Bot):
    file_id = msg.voice.file_id
    file = await bot.get_file(file_id)
    await msg.answer(text.wait_feedback)
    audio: io.BytesIO = await bot.download_file(file.file_path)
    y, sr = librosa.load(audio, sr=16000)
    probs = emotion_recognizer.recognize(y, sr)

    await msg.answer(text.result.format(*probs))
    await msg.answer(text.ask_continue, reply_markup=kb.continue_kb)


@router.message(Intonation.angry)
async def angry_other(msg: Message):
    await msg.answer(text.angry_instruction, reply_markup=kb.continue_kb)
