"""
Вспомогательные классы для FSM (машины состояний)
"""

from aiogram.fsm.state import StatesGroup, State


class Intonation(StatesGroup):
    """Состояния-режимы тренировки
    (menu - состояние без запущенных тренировок)."""
    happy = State()
    sad = State()
    angry = State()
    menu = State()
