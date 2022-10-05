import os
import pathlib
import uuid
import whisper
from telebot import TeleBot
from telebot.types import Message


class WhisperSpeechBot:
    def __init__(self):
        self.basement = pathlib.Path(__file__).parent.absolute()
        self.upload_dir = self.basement / "uploads"
        self.upload_dir.mkdir(exist_ok=True)

        self.bot = TeleBot(os.environ["BOT_TOKEN"])

        self.model = whisper.load_model("small", device="cuda")

        self.listeners()

    def download_telegram_file(self, message, file_type):
        # download file by choosing content_type
        if message.content_type == "voice":
            voice_file = self.bot.get_file(message.voice.file_id)
            cached_file = self.bot.download_file(voice_file.file_path)
            filename = f"{message.chat.id}_{str(uuid.uuid4())}.ogg"

        elif message.content_type == "video":
            video_file = self.bot.get_file(message.video.file_id)
            cached_file = self.bot.download_file(video_file.file_path)
            filename = f"{message.chat.id}_{str(uuid.uuid4())}.mp4"

        elif message.content_type == "audio":
            audio_file = self.bot.get_file(message.audio.file_id)
            cached_file = self.bot.download_file(audio_file.file_path)
            filename = f"{message.chat.id}_{str(uuid.uuid4())}.wav"

        elif message.content_type == "video_note":
            video_note_file = self.bot.get_file(message.video_note.file_id)
            cached_file = self.bot.download_file(video_note_file.file_path)
            filename = f"{message.chat.id}_{str(uuid.uuid4())}.mp4"

        else:
            raise Exception("Unknown content_type")

        filename_path = self.upload_dir / filename

        with filename_path.open("wb") as new_file:
            new_file.write(cached_file)

        return filename, filename_path

    def listeners(self):
        @self.bot.message_handler(content_types=["audio", "voice", "video", "video_note"])
        def translate(message: Message):
            new_message = self.bot.reply_to(message, "Обрабатываю нах...")

            filename, filename_path = self.download_telegram_file(message, message.content_type)
            result = self.model.transcribe(str(filename_path))

            if not result["text"]:
                return self.bot.edit_message_text(
                    message_id=new_message.id,
                    chat_id=new_message.chat.id,
                    text="Че ты за говно отправил? Не слышу нихуя"
                )

            self.bot.edit_message_text(
                message_id=new_message.id,
                chat_id=new_message.chat.id,
                text=result["text"]
            )

    def run(self):
        print("Running bot...")
        self.bot.polling(none_stop=True, timeout=9999999)


bot = WhisperSpeechBot()
bot.run()
