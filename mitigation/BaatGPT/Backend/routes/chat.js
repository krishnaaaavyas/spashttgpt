// Backend/routes/chat.js
import express from 'express';
import axios from 'axios';
import { generateText } from '../utils/openai.js';

const router = express.Router();
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:5000';

// Health proxy to the ML service
router.get('/ml/health', async (req, res) => {
  try {
    const r = await axios.get(`${ML_SERVICE_URL}/health`, { timeout: 5000 });
    res.json({ ml_service_status: 'connected', ...r.data });
  } catch (e) {
    res.status(503).json({ ml_service_status: 'disconnected', error: e.message });
  }
});

// User input -> OpenAI -> MedBERT risk
router.post('/thread', async (req, res) => {
  try {
    const { prompt } = req.body;
    if (!prompt) return res.status(400).json({ success: false, error: 'prompt is required' });

    // 1) Generate text with OpenAI
    const openaiOutput = await generateText(prompt);

    if (!openaiOutput) {
      return res.status(400).json({ success: false, error: 'Failed to generate text from OpenAI' });
    }

    // 2) Send generated text to the Flask MedBERT service
    const mlResp = await axios.post(
      `${ML_SERVICE_URL}/predict`,
      { text: openaiOutput },
      { timeout: 30000 }
    );

    // 3) Return a consolidated response
    return res.json({
      success: true,
      openai_output: openaiOutput,
      risk_prediction: mlResp.data.prediction,
      risk_confidence: mlResp.data.confidence,
      probabilities: mlResp.data.probabilities
    });
  } catch (err) {
    console.error('pipeline error:', err?.response?.data || err.message);
    return res.status(502).json({
      success: false,
      error: 'Generation or risk scoring failed'
    });
  }
});

// Streaming variant: send OpenAI output first, then ML flags when ready
router.post('/thread/stream', async (req, res) => {
  // SSE-like chunked responses
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders && res.flushHeaders();

  try {
    const { prompt } = req.body;
    if (!prompt) {
      res.write(`data: ${JSON.stringify({ type: 'error', message: 'prompt is required' })}\n\n`);
      return res.end();
    }

    // 1) Generate text with OpenAI
    const openaiOutput = await generateText(prompt);
    if (!openaiOutput) {
      res.write(`data: ${JSON.stringify({ type: 'error', message: 'OpenAI generation failed' })}\n\n`);
      return res.end();
    }

    // send OpenAI output immediately
    res.write(`data: ${JSON.stringify({ type: 'openai', text: openaiOutput })}\n\n`);

    // 2) Call ML service for risk scoring
    const mlResp = await axios.post(
      `${ML_SERVICE_URL}/predict`,
      { text: openaiOutput },
      { timeout: 30000 }
    );

    // send ML flags
    res.write(
      `data: ${JSON.stringify({
        type: 'ml',
        prediction: mlResp.data.prediction,
        confidence: mlResp.data.confidence,
        probabilities: mlResp.data.probabilities
      })}\n\n`
    );

    // done
    res.write(`data: ${JSON.stringify({ type: 'done' })}\n\n`);
    return res.end();
  } catch (err) {
    console.error('stream pipeline error:', err?.response?.data || err?.message || err);
    res.write(`data: ${JSON.stringify({ type: 'error', message: 'Generation or scoring failed' })}\n\n`);
    return res.end();
  }
});

export default router;
