const { headers } = require("../data/global");

const apiKey = "AIzaSyASJvnbHLrtr_w6XjU_qTcLErXKtTk33g4";
// The path to the remote audio file.
const gcsUri = "gs://ibm_watsonx/audio-files/1719789028873_yls2lliqr.wav";

async function transcribeSpeech() {
    const config = {
        model: "latest_long",
        encoding: "OGG_OPUS",
        sampleRateHertz: 48000,
        audioChannelCount: 2,
        enableWordTimeOffsets: true,
        enableWordConfidence: true,
        languageCode: "ml-IN",
    };

    const request = {
        audio: {
            uri: gcsUri,
        },
        config: config,
    };

    try {
        const response = await fetch(`https://speech.googleapis.com/v1/speech:longrunningrecognize?key=${apiKey}`, {
            method: 'POST',
            headers: {
            'Content-Type': 'application/json',
            },
            body: JSON.stringify(request),
        });
        console.log('Response:', response.json);
        const operationName = response.data.name;

        // Poll the operation status until it's done
        let operationResponse;
        do {
            const operationResponse = await fetch(`https://speech.googleapis.com/v1/operations/${operationName}?key=${apiKey}`, {
                headers: {
                    'Content-Type': 'application/json',
                },
            });

            if (operationResponse.data.done) {
                break;
            }

            await new Promise(resolve => setTimeout(resolve, 1000)); // Wait for 1 second before polling again
        } while (true);

        const transcription = operationResponse.data.response.results
            .map(result => result.alternatives[0].transcript)
            .join('\n');
        console.log(`Transcription: ${transcription}`);
    } catch (error) {
        console.error('Error:', error);
    }
}

transcribeSpeech();