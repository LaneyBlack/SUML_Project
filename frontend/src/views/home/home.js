import React, {useState} from 'react'
import {Helmet} from 'react-helmet'
// Relative imports
import './home.css'
import {backendService} from "../../components/backendService";

const Home = (props) => {
    const [title, setTitle] = useState("");
    const handleTitle = (event) => {
        setTitle(event.target.value); // Update the state with the new value
    };
    const [body, setBody] = useState("");
    const handleBody = (event) => {
        setBody(event.target.value); // Update the state with the new value
    };
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState("");

    const handlePredict = async () => {
        setError("");
        setResult(null);

        if (!title.trim()) {
            setError("* Oops! Please add a title to your news story before proceeding.");
            return;
        }
        if (!body.trim()) {
            setError("* Oops! The news body is empty. Please add some text before proceeding.");
            return;
        }
        if (body.length < 100) {
            setError("* Text is too short. Please provide at least 100 characters in the body.");
            return;
        }
        if (body.length > 10000) {
            setError("* Text is too long. Please limit your input to 10,000 characters.");
            return;
        }

        setLoading(true);

        try {
            const response = await backendService.predict(title, body);

            if (response.error) {
                setError(`* Error: ${response.error}`);
            } else {
                console.log(response)
                setResult({
                    message: response.message, // Displaying the success message
                    prediction: response.prediction.label, // "REAL" or "FAKE"
                    precision: response.prediction.confidence, // Confidence as a percentage
                });
            }
        } catch (err) {
            setError("* An unexpected error occurred. Please try again.");
        } finally {
            setLoading(false);
        }
    };


    return (<div className="main-container">
        <Helmet>
            <title>Fake News Detector</title>
            <meta property="og:title" content="Fake News Detector"/>
        </Helmet>
        <div className="home-main">
            <img alt="image"
                 src="https://news.stanford.edu/__data/assets/image/0021/41808/Fake-news-and-facts-digital-concept.jpeg"
                 className="home-image"/>
            <h1 className="main-title">Fake News Detection</h1>
            {/*<Link to="#user-manual" smooth>User manual</Link>*/}
            <a href="/#user-manual" className="user-manual-link">User manual</a>
            <div className="home-input">
                    <span className="home-text-past">
                        Paste the news into the text field 📰
                    </span>
                <textarea
                    placeholder="Title of Your news .."
                    className="textarea-name textarea"
                    value={title}
                    onChange={handleTitle}
                >
                </textarea>
                <textarea
                    placeholder="Body of your news .."
                    className="textarea-body textarea"
                    value={body}
                    onChange={handleBody}
                >
                </textarea>

                {/* Display error messages */}
                {error && <span className="error-text">{error}</span>}

                {/*Buttons*/}
                <div className="buttons-container">
                    <button
                        type="button"
                        className="reset-button button"
                        onClick={() => {
                            setTitle("");
                            setBody("");
                            setError("");
                            setResult(null);
                        }}>
                        Reset
                    </button>
                    <button
                        type="button"
                        className="predict-button button"
                        onClick={handlePredict}
                        disabled={loading}
                    >
                        Predict
                    </button>
                </div>
            </div>

            {/*---Loading animation---*/}
            {loading && (
                <svg width="24" height="24" viewBox="0 0 24 24" className="home-icon1">
                    <circle r="0" cx="18" cy="12" fill="currentColor">
                        <animate
                            dur="1.5s"
                            begin=".67"
                            values="0;2;0;0"
                            calcMode="spline"
                            keySplines="0.2 0.2 0.4 0.8;0.2 0.2 0.4 0.8;0.2 0.2 0.4 0.8"
                            repeatCount="indefinite"
                            attributeName="r">
                        </animate>
                    </circle>
                    <circle r="0" cx="12" cy="12" fill="currentColor">
                        <animate
                            dur="1.5s"
                            begin=".33"
                            values="0;2;0;0"
                            calcMode="spline"
                            keySplines="0.2 0.2 0.4 0.8;0.2 0.2 0.4 0.8;0.2 0.2 0.4 0.8"
                            repeatCount="indefinite"
                            attributeName="r">
                        </animate>
                    </circle>
                    <circle r="0" cx="6" cy="12" fill="currentColor">
                        <animate
                            dur="1.5s"
                            begin="0"
                            values="0;2;0;0"
                            calcMode="spline"
                            keySplines="0.2 0.2 0.4 0.8;0.2 0.2 0.4 0.8;0.2 0.2 0.4 0.8"
                            repeatCount="indefinite"
                            attributeName="r"
                        ></animate>
                    </circle>
                </svg>
            )}

            {/*---Output---*/}
            {result && result.prediction === "FAKE" && (
                <div className="home-output-fake1">
                    <span className="fake-output-text">FAKE</span>
                    <span className="fake-output-text-precision">
                        🎯 Precision: {result.precision}%
                    </span>
                </div>
            )}

            {result && result.prediction === "REAL" && (
                <div className="home-output-fact">
                    <span className="fact-output-text">FACT</span>
                    <span className="fact-output-text-precision">
                    🎯 Precision: {result.precision}%
                    </span>
                </div>
            )}
            {/*---User manual---*/}
            <div id="user-manual" className="home-user-manual">
                <h1 className="text-welcome-instruction">
                    Welcome to the Fake News Detection Tool!
                </h1>
                <span className="text-intro">
                    Whether you’re here to bust some fake news or accidentally train
                    your own model into thinking everything is a conspiracy (please
                    don't), this guide will walk you through every step.
                    Let's get started!
                    <br></br>
                    <br></br>
                </span>
                <div className="instructions-container">
                    <span className="instructions-title">
                      <span>Step 1: Input Some Text</span>
                    </span>
                    <span className="instruction-text">
                        <span>
                            Go to the input box. (You can’t miss it—it’s big and empty, like
                            your calendar on a Friday night.)
                        </span>
                        <br></br>
                        <span>
                            Copy-paste a news article, type in a headline, or invent your
                            own conspiracy theory. The AI won’t judge... much.
                        </span>
                    </span>
                    <span className="instructions-title">
                        <span>Step 2: Pick an Action</span>
                    </span>
                    <span className="instruction-text">
                        A) Reset (For Mistakes): If you change your mind, click the
                        Reset button. The text box will clear out, as if nothing ever
                        happened.  Perfect for “Wait, this isn 't fake news—it’s my
                        to-do list!” moments.
                    </span>
                    <span className="instruction-text">
                        <span>
                        B) Train (Teach the AI): Click the Train button. A new popup
                        will appear asking for your input:Is the text you entered Fact
                        or Fake? Select one. The AI will process your choice and either
                        celebrate your success with a cheerful “Training Successful!”
                        message or throw an “Error” if something went wrong (like trying
                        to outsmart it by labeling pizza recipes as fake news).
                        Congratulations! You’ve just helped train the AI to be smarter.
                        Gold star for you!
                        </span>
                        <br></br>
                    </span>
                    <span className="instruction-text">
                        C) Predict (Time for Judgment): Click the Predict button. The AI
                        will evaluate your text like a judge on a talent show. It will
                        tell you whether the text is likely true or fake. You’ll also
                        get a confidence percentage because even robots like to show
                        their work. Example: “This text is 92% likely to be fake, and 8%
                        likely to be true. Sorry, not sorry.”
                    </span>
                    <span className="text-intro">
                        <span>
                            Congratulations! You now know how to use the Fake News Detection tool like a pro.
                        Whether you’re fighting misinformation, training a
                        hyper-intelligent robot, or just here for the laughs, we hope
                        you enjoy the ride.
                        </span>
                        <br></br>
                        <br></br>
                        <span>
                            Remember: Fake news is like bad karaoke—funny at first, but
                            harmful if taken seriously.
                        </span>
                        <br></br>
                        <br></br>
                        <span>Now, go forth and detect!</span>
                    </span>
                </div>
            </div>
            <span>
                <span className="made-by">Made by Chrzczone Chrząszcze Team </span>
            </span>
        </div>
    </div>)
}

export default Home
