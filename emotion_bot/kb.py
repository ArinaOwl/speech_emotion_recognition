from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, KeyboardButton, ReplyKeyboardMarkup

menu = [
    [InlineKeyboardButton(text="😄 Радость", callback_data="happy"),
     InlineKeyboardButton(text="😔 Грусть", callback_data="sad"),
     InlineKeyboardButton(text="😠 Злость", callback_data="angry")],
    [InlineKeyboardButton(text="🔎 Помощь", callback_data="help")]
]
start_exit = [
    [InlineKeyboardButton(text="Начать", callback_data="next_phrase")]
]
continue_exit = [
    [InlineKeyboardButton(text="Продолжить", callback_data="next_phrase")]
]
menu = InlineKeyboardMarkup(inline_keyboard=menu)
start_kb = InlineKeyboardMarkup(inline_keyboard=start_exit)
continue_kb = InlineKeyboardMarkup(inline_keyboard=continue_exit)
exit_kb = ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text="◀️ Выйти в меню")]], resize_keyboard=True)
