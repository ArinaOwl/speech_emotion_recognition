"""
Вспомогательные классы для FSM (машины состояний)
"""

from aiogram.fsm.state import StatesGroup, State


class Intonation(StatesGroup):
    happy = State()
    sad = State()
    angry = State()
    menu = State()
