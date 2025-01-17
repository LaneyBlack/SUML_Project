import React from 'react';
import ReactDOM from 'react-dom/client'; // Use react-dom/client instead of react-dom
import App from './App.js';
import reportWebVitals from "./reportWebVitals";

const rootElement = document.getElementById('app');

// Create a root and render the App component
const root = ReactDOM.createRoot(rootElement);
root.render(
    <React.StrictMode>
        <App />
    </React.StrictMode>
);

// Measuring performance
reportWebVitals(console.log);