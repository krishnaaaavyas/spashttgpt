// Backend/utils/openai.js
import OpenAI from 'openai';
import dotenv from 'dotenv';

// Load environment variables from .env file
dotenv.config({ path: '../../.env' });

if (!process.env.OPENAI_API_KEY) {
  console.warn('⚠️ OpenAI API key not found in environment variables. OpenAI calls will fail until a key is set.');
}

const openai = process.env.OPENAI_API_KEY ? new OpenAI({ apiKey: process.env.OPENAI_API_KEY }) : null;

// Generate text with OpenAI Chat API
export async function generateText(prompt, model = process.env.OPENAI_MODEL || 'gpt-3.5-turbo') {
  try {
    if (!openai) throw new Error('OpenAI API key not configured');
    const completion = await openai.chat.completions.create({
      model: model,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.7,
    });
    return completion.choices[0].message.content;
  } catch (error) {
    console.error('OpenAI API Error:', error);
    throw new Error('Failed to generate text from OpenAI');
  }
}

export default { generateText };
