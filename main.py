from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton
from aiogram.filters import Command
import asyncio
import torch
from predict import predict_emotion
from dotenv import load_dotenv
import os

model = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pretrained_model/resnet50_FER2013+_full_model'))

model.eval()

load_dotenv()
token = os.getenv('TOKEN')

bot = Bot(token)
dp = Dispatcher()

@dp.message(Command("start"))
async def start_command(message: Message):
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text='Start')],
            [KeyboardButton(text='Help')],
            [KeyboardButton(text='About')]
        ],
        resize_keyboard=True
    )
    await message.answer("Выберите действие:", reply_markup=keyboard)

@dp.message(F.content_type == 'text')
async def handle_button(message: Message):
    if message.text == "Start":
        response_text = "Привет! Я бот для определения эмоций на лице. Отправь мне фото, и я скажу, что ты чувствуешь!"
    elif message.text == "Help":
        response_text = "Я могу распознавать эмоции на вашем лице. Просто отправьте мне фото лица. Команды:\nStart - Начать\nHelp - Помощь\nAbout - О боте"
    elif message.text == "About":
        response_text = "Этот бот использует модель ResNet50, обученную на наборе данных FER2013+ для распознавания эмоций."
    else:
        response_text = "Команда не распознана. Попробуйте снова."
    await message.answer(response_text)

@dp.message(F.content_type == 'photo')
async def handle_photo(message: Message):
    file_path = "temp_photo.jpg"
    try:
        photo = message.photo[-1].file_id
        file_info = await bot.get_file(photo)

        await bot.download_file(file_info.file_path, destination=file_path)

        emotion = predict_emotion(file_path, model)
        if not emotion:
            raise ValueError("Модель не вернула результат.")

        await message.reply(f"Эмоция на лице: {emotion}")
    except Exception as e:
        print(f"Ошибка анализа: {e}")
        await message.reply("Ошибка при анализе фото. Попробуйте снова!")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

async def main():
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())