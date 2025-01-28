const BASE_URL = 'https://suml-project.onrender.com'; // Base URL for the server
// const BASE_URL = 'http://127.0.0.1:8000'; // Base URL for the local

export const backendService = {
    predict: async (title, text) => {
        try {
            // Make a request
            const response = await fetch(`${BASE_URL}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    title: title,
                    text: text,
                })
            });
            // Resolve the response
            if (!response.ok) {
                const errorData = await response.json();
                console.log(errorData);
                return {error: errorData.error || 'Unexpected format'};
            }

            const data = await response.json();
            return data ? data : {error: 'Unexpected format'};
        } catch (error) {
            // console.error('Error during action:', body.action, error);
            return {error: 'Network or server error'};
        }
    },

    fineTune: async (title, text, label) => {
        try {
            // Make a request
            const response = await fetch(`${BASE_URL}/fine-tune`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    title: title,
                    text: text,
                    label: label
                })
            });
            // Resolve the response
            if (!response.ok) {
                const errorData = await response.json();
                return {error: errorData.error || 'Unexpected format'};
            }

            const data = await response.json();
            return data ? data : {error: 'Unexpected format'};
        } catch (error) {
            // console.error('Error during action:', body.action, error);
            return {error: 'Network or server error'};
        }
    },
};