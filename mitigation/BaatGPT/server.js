// server.js (root of BaatGPT)
import express from 'express';
import bodyParser from 'body-parser';
import cors from 'cors';
import mongoose from 'mongoose';
import 'dotenv/config';
import axios from 'axios';

import chatRoutes from './Backend/routes/chat.js';

const app = express();
const PORT = process.env.PORT || 3000;  // Changed to 3000 to avoid conflict with ML service
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:5000';

// Configure CORS with specific options
// Allow CORS from local dev servers (any localhost port) and common local origins
app.use(cors({
  origin: [/^http:\/\/localhost(:\d+)?$/, /^http:\/\/127\.0\.0\.1(:\d+)?$/],
  methods: ['GET', 'POST', 'OPTIONS', 'PUT', 'DELETE'],
  credentials: true,
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With'],
  optionsSuccessStatus: 200
}));

// Ensure preflight (OPTIONS) requests receive CORS headers
app.options('*', cors());
app.use(bodyParser.json());

// Optional MongoDB connection (skip if not provided)
const MONGODB_URI = process.env.MONGODB_URI;
async function startServer() {
  try {
    if (MONGODB_URI) {
      await mongoose.connect(MONGODB_URI);
      console.log('MongoDB connected successfully');
    } else {
      console.log('No MONGODB_URI provided; skipping MongoDB connection');
    }

    // Mount API routes
    app.use('/api', chatRoutes);  // This mounts all chat routes under /api prefix
    
    // Basic root status
    app.get('/', (_req, res) => res.json({ ok: true, api: '/api', ml: ML_SERVICE_URL }));

    app.listen(PORT, () => {
      console.log(`🚀 Server running on port ${PORT}`);
      console.log(`📊 ML Service URL: ${ML_SERVICE_URL}`);
    });
  } catch (err) {
    console.error('Startup error:', err.message);
    process.exit(1);
  }
}

startServer();
