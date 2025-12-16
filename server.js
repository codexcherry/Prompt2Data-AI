const express = require('express');
const multer = require('multer');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const { GoogleGenerativeAI } = require('@google/generative-ai');

// Load environment variables
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

// Initialize Gemini AI
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.static('public'));

// Multer configuration for file uploads
const storage = multer.memoryStorage();
const upload = multer({
    storage: storage,
    limits: { fileSize: 10 * 1024 * 1024 } // 10MB limit
});

// Ensure uploads directory exists
if (!fs.existsSync('uploads')) {
    fs.mkdirSync('uploads');
}

// Parse different file types
function parseFileContent(buffer, filename) {
    const ext = path.extname(filename).toLowerCase();
    const content = buffer.toString('utf-8');

    try {
        if (ext === '.json') {
            return { type: 'json', data: JSON.parse(content), raw: content };
        } else if (ext === '.csv') {
            return { type: 'csv', data: parseCSV(content), raw: content };
        } else {
            // Treat as text/other
            return { type: 'text', data: content, raw: content };
        }
    } catch (error) {
        return { type: 'text', data: content, raw: content };
    }
}

// Simple CSV parser
function parseCSV(content) {
    const lines = content.trim().split('\n');
    if (lines.length === 0) return [];

    const headers = parseCSVLine(lines[0]);
    const data = [];

    for (let i = 1; i < lines.length; i++) {
        const values = parseCSVLine(lines[i]);
        const row = {};
        headers.forEach((header, index) => {
            row[header.trim()] = values[index]?.trim() || '';
        });
        data.push(row);
    }

    return data;
}

// Parse CSV line handling quoted values
function parseCSVLine(line) {
    const result = [];
    let current = '';
    let inQuotes = false;

    for (let i = 0; i < line.length; i++) {
        const char = line[i];

        if (char === '"') {
            inQuotes = !inQuotes;
        } else if (char === ',' && !inQuotes) {
            result.push(current);
            current = '';
        } else {
            current += char;
        }
    }
    result.push(current);

    return result;
}

// Convert data back to CSV
function dataToCSV(data) {
    if (!Array.isArray(data) || data.length === 0) return '';

    const headers = Object.keys(data[0]);
    const csvLines = [headers.join(',')];

    for (const row of data) {
        const values = headers.map(h => {
            const val = String(row[h] || '');
            // Escape values with commas or quotes
            if (val.includes(',') || val.includes('"') || val.includes('\n')) {
                return `"${val.replace(/"/g, '""')}"`;
            }
            return val;
        });
        csvLines.push(values.join(','));
    }

    return csvLines.join('\n');
}

// Upload and process endpoint
app.post('/api/upload', upload.single('file'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No file uploaded' });
        }

        const parsed = parseFileContent(req.file.buffer, req.file.originalname);

        res.json({
            success: true,
            filename: req.file.originalname,
            type: parsed.type,
            data: parsed.data,
            preview: Array.isArray(parsed.data) ? parsed.data.slice(0, 10) : parsed.data
        });
    } catch (error) {
        console.error('Upload error:', error);
        res.status(500).json({ error: 'Failed to process file: ' + error.message });
    }
});

// Process with AI endpoint
app.post('/api/process', async (req, res) => {
    try {
        const { data, prompt, type } = req.body;

        if (!data || !prompt) {
            return res.status(400).json({ error: 'Data and prompt are required' });
        }

        if (!process.env.GEMINI_API_KEY || process.env.GEMINI_API_KEY === 'your_gemini_api_key_here') {
            return res.status(400).json({ error: 'Please configure your Gemini API key in .env file' });
        }

        const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });

        // Prepare data string for AI
        let dataString;
        if (type === 'csv' || Array.isArray(data)) {
            dataString = JSON.stringify(data, null, 2);
        } else if (type === 'json') {
            dataString = JSON.stringify(data, null, 2);
        } else {
            dataString = data;
        }

        // Craft the AI prompt
        const aiPrompt = `You are a data preprocessing assistant. You will receive a dataset and a user instruction.
Your task is to transform/preprocess the data according to the user's instruction.

CRITICAL JSON FORMATTING RULES:
1. Return ONLY valid JSON - no explanations, no markdown, no code blocks, no extra text
2. ALL keys and string values MUST be properly quoted with double quotes
3. Escape all special characters in strings: use \\" for quotes, \\n for newlines, \\\\ for backslashes
4. Column names with special characters (spaces, brackets, dashes) MUST be quoted: "Column Name"
5. If the input is an array of objects, return an array of objects: [{...}, {...}]
6. If the input is a single object, return an object: {...}
7. If the input is text, return: {"data": "processed text"}
8. Preserve the structure as much as possible unless the user asks to change it
9. Handle missing values, duplicates, type conversions as requested
10. Ensure all JSON is valid and parseable - test your output mentally before responding

EXAMPLE VALID OUTPUT:
[{"Name": "John", "Age": 25, "AL405- [P]": "A+"}, {"Name": "Jane", "Age": 30, "AL405- [P]": "B"}]

USER'S INSTRUCTION: ${prompt}

DATASET TO PROCESS:
${dataString}

RESPOND WITH ONLY THE PROCESSED JSON DATA, NO OTHER TEXT, NO EXPLANATIONS:`;

        // Add retry logic for quota and service unavailable issues
        let result, response, responseText;
        let retries = 3;
        let retryDelay = 2000; // Start with 2 seconds
        
        while (retries > 0) {
            try {
                result = await model.generateContent(aiPrompt);
                response = await result.response;
                responseText = response.text().trim();
                break; // Success, exit retry loop
            } catch (retryError) {
                retries--;
                const errorMsg = retryError.message || '';
                const errorStatus = retryError.status || '';
                const errorStatusText = retryError.statusText || '';
                
                // Check if error is retryable
                const shouldRetry = (
                    (errorMsg.includes('quota') || 
                     errorMsg.includes('overloaded') || 
                     errorMsg.includes('503') ||
                     errorMsg.includes('Service Unavailable') ||
                     errorStatus === 503 ||
                     errorStatusText.includes('Service Unavailable')) && 
                    retries > 0
                );
                
                if (shouldRetry) {
                    console.log(`API error (${errorMsg.substring(0, 50)}...), retrying in ${retryDelay/1000} seconds... (${retries} retries left)`);
                    await new Promise(resolve => setTimeout(resolve, retryDelay));
                    retryDelay *= 2; // Exponential backoff: 2s, 4s, 8s
                } else {
                    throw retryError; // Re-throw if not retryable error or no retries left
                }
            }
        }
        
        // Check if we got a response after retries
        if (!responseText) {
            throw new Error('Failed to get response from AI after retries');
        }

        // Clean up response - remove markdown code blocks if present
        responseText = responseText.replace(/```json\s*/gi, '').replace(/```\s*/g, '').trim();

        // Try to parse the response with better error handling
        let processedData;
        try {
            processedData = JSON.parse(responseText);
        } catch (parseError) {
            console.error('JSON parse error:', parseError.message);
            console.error('Response text (first 500 chars):', responseText.substring(0, 500));
            
            // Try to extract JSON array or object first
            const jsonMatch = responseText.match(/\[[\s\S]*\]|\{[\s\S]*\}/);
            if (jsonMatch) {
                let cleanedJson = jsonMatch[0];
                
                try {
                    // First attempt: parse as-is
                    processedData = JSON.parse(cleanedJson);
                } catch (secondParseError) {
                    console.error('First parse attempt failed:', secondParseError.message);
                    
                    // Second attempt: Fix common issues
                    // Fix unescaped quotes in string values
                    cleanedJson = cleanedJson.replace(/: "([^"]*)"([^",}\]]*)"([^",}\]]*)"/g, (match, p1, p2, p3) => {
                        // If there are multiple quotes in a value, escape them
                        return `: "${p1}\\"${p2}\\"${p3}"`;
                    });
                    
                    // Fix unquoted keys with special characters
                    cleanedJson = cleanedJson.replace(/([{,]\s*)([A-Za-z0-9_\-\[\] ]+)(\s*:)/g, (match, prefix, key, suffix) => {
                        // If key has spaces, brackets, or dashes and isn't quoted, quote it
                        if (!key.startsWith('"') && (key.includes(' ') || key.includes('-') || key.includes('[') || key.includes(']'))) {
                            return `${prefix}"${key.replace(/"/g, '\\"')}"${suffix}`;
                        }
                        return match;
                    });
                    
                    try {
                        processedData = JSON.parse(cleanedJson);
                    } catch (thirdParseError) {
                        console.error('Second parse attempt failed:', thirdParseError.message);
                        console.error('Problematic JSON (first 500 chars):', cleanedJson.substring(0, 500));
                        
                        // Third attempt: More aggressive cleaning
                        // Remove any text before first [ or {
                        const firstBrace = cleanedJson.indexOf('[');
                        const firstCurly = cleanedJson.indexOf('{');
                        let startIndex = -1;
                        if (firstBrace !== -1 && firstCurly !== -1) {
                            startIndex = Math.min(firstBrace, firstCurly);
                        } else if (firstBrace !== -1) {
                            startIndex = firstBrace;
                        } else if (firstCurly !== -1) {
                            startIndex = firstCurly;
                        }
                        
                        if (startIndex !== -1) {
                            cleanedJson = cleanedJson.substring(startIndex);
                            // Find matching closing bracket/brace
                            const closingBrace = cleanedJson.lastIndexOf(']');
                            const closingCurly = cleanedJson.lastIndexOf('}');
                            let endIndex = -1;
                            if (closingBrace !== -1 && closingCurly !== -1) {
                                endIndex = Math.max(closingBrace, closingCurly) + 1;
                            } else if (closingBrace !== -1) {
                                endIndex = closingBrace + 1;
                            } else if (closingCurly !== -1) {
                                endIndex = closingCurly + 1;
                            }
                            
                            if (endIndex !== -1) {
                                cleanedJson = cleanedJson.substring(0, endIndex);
                            }
                        }
                        
                        try {
                            processedData = JSON.parse(cleanedJson);
                        } catch (fourthParseError) {
                            console.error('All parse attempts failed:', fourthParseError.message);
                            // Last resort: return error with helpful message
                            return res.status(400).json({ 
                                error: 'AI returned invalid JSON. The response contains malformed JSON that cannot be parsed. Please try again with a simpler transformation request.',
                                details: parseError.message,
                                suggestion: 'Try breaking your request into smaller steps or rephrasing it.',
                                responsePreview: responseText.substring(0, 300)
                            });
                        }
                    }
                }
            } else {
                // No JSON found, return as text wrapper
                console.warn('No JSON structure found in AI response, returning as text');
                processedData = { data: responseText };
            }
        }

        res.json({
            success: true,
            data: processedData,
            originalType: type
        });

    } catch (error) {
        console.error('Processing error:', error);

        // More specific error messages
        if (error.message.includes('API key')) {
            res.status(400).json({ error: 'Invalid API key. Please check your Gemini API key in .env file' });
        } else if (error.message.includes('quota')) {
            res.status(429).json({ error: 'API quota exceeded. Please try again later' });
        } else if (error.message.includes('model')) {
            res.status(400).json({ error: 'Model not available. Please try again or contact support' });
        } else {
            res.status(500).json({ error: 'AI processing failed: ' + error.message });
        }
    }
});

// Export endpoint
app.post('/api/export', (req, res) => {
    try {
        const { data, format } = req.body;

        let content, contentType, filename;

        if (format === 'csv') {
            content = dataToCSV(Array.isArray(data) ? data : [data]);
            contentType = 'text/csv';
            filename = 'processed_data.csv';
        } else if (format === 'json') {
            content = JSON.stringify(data, null, 2);
            contentType = 'application/json';
            filename = 'processed_data.json';
        } else {
            content = typeof data === 'string' ? data : JSON.stringify(data, null, 2);
            contentType = 'text/plain';
            filename = 'processed_data.txt';
        }

        res.json({
            success: true,
            content: content,
            contentType: contentType,
            filename: filename
        });

    } catch (error) {
        console.error('Export error:', error);
        res.status(500).json({ error: 'Export failed: ' + error.message });
    }
});

// ML Training endpoint
app.post('/api/train', async (req, res) => {
    try {
        const { data, targetColumn, models, trainTestSplit } = req.body;

        // Validate request
        if (!data || !Array.isArray(data) || data.length === 0) {
            return res.status(400).json({ error: 'Valid data array is required' });
        }

        // Target column is optional - if not provided, will assess data quality
        // if (!targetColumn) {
        //     return res.status(400).json({ error: 'Target column is required' });
        // }

        if (!models || !Array.isArray(models) || models.length === 0) {
            return res.status(400).json({ error: 'At least one model must be selected' });
        }

        // Prepare request for Python service
        const pythonRequest = {
            data: data,
            targetColumn: targetColumn,
            models: models,
            trainTestSplit: trainTestSplit || 0.8
        };

        // Spawn Python process
        const { spawn } = require('child_process');
        const pythonProcess = spawn('python', ['ml_service.py']);

        let stdout = '';
        let stderr = '';

        // Send data to Python process
        pythonProcess.stdin.write(JSON.stringify(pythonRequest));
        pythonProcess.stdin.end();

        // Collect stdout
        pythonProcess.stdout.on('data', (data) => {
            stdout += data.toString();
        });

        // Collect stderr
        pythonProcess.stderr.on('data', (data) => {
            stderr += data.toString();
        });

        // Handle process completion
        pythonProcess.on('close', (code) => {
            if (code !== 0) {
                console.error('Python process error:', stderr);
                return res.status(500).json({ 
                    error: 'ML training failed: ' + (stderr || 'Unknown error'),
                    success: false
                });
            }

            try {
                // Clean up NaN values before parsing
                let cleanedOutput = stdout.trim();
                // Replace NaN with null (valid JSON)
                cleanedOutput = cleanedOutput.replace(/NaN/g, 'null');
                // Replace Infinity with null
                cleanedOutput = cleanedOutput.replace(/Infinity/g, 'null');
                cleanedOutput = cleanedOutput.replace(/-Infinity/g, 'null');
                
                const result = JSON.parse(cleanedOutput);
                
                if (!result.success) {
                    return res.status(400).json({
                        error: result.error || 'Training failed',
                        success: false
                    });
                }

                res.json(result);
            } catch (parseError) {
                console.error('Failed to parse Python output:', stdout);
                console.error('Parse error:', parseError.message);
                res.status(500).json({ 
                    error: 'Failed to parse training results: ' + parseError.message,
                    success: false,
                    rawOutput: stdout.substring(0, 500) // First 500 chars for debugging
                });
            }
        });

        // Handle process errors
        pythonProcess.on('error', (error) => {
            console.error('Failed to start Python process:', error);
            res.status(500).json({ 
                error: 'Failed to start ML service. Make sure Python is installed with required packages (scikit-learn, pandas, numpy).',
                success: false
            });
        });

    } catch (error) {
        console.error('Training error:', error);
        res.status(500).json({ error: 'Training failed: ' + error.message, success: false });
    }
});

// Health check
app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.listen(PORT, () => {
    console.log(`🚀 Dataset Preprocessor running at http://localhost:${PORT}`);
    console.log(`📁 Upload your dataset and use AI to preprocess it!`);
});
