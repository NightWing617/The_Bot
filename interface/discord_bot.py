
# discord_bot.py

import discord
import asyncio
from betting.kelly_calculator import calculate_kelly_bets
from explainability.nlp_betting_summary import generate_summary

TOKEN = "your_discord_bot_token"

class RaceBot(discord.Client):
    async def on_ready(self):
        print(f"Logged in as {self.user}")

    async def on_message(self, message):
        if message.author == self.user:
            return

        if message.content.startswith("!bets"):
            # Dummy predictions for demo
            predictions = [
                {'horse': 'Lightning Bolt', 'probability': 0.35, 'odds': 4.0},
                {'horse': 'Dark Shadow', 'probability': 0.25, 'odds': 6.0}
            ]
            bets = calculate_kelly_bets(predictions)
            summary = generate_summary(bets)
            await message.channel.send("```" + summary + "```")

client = RaceBot(intents=discord.Intents.default())
client.run(TOKEN)
