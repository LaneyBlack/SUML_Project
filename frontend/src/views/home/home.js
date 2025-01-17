import React from 'react'

import {Helmet} from 'react-helmet'

import './home.css'

const Home = (props) => {
    return (<div className="home-container1">
        <Helmet>
            <title>Face News Detector</title>
            <meta property="og:title" content="Face News Detector"/>
        </Helmet>
        <div className="home-main">
            <img alt="image"
                src="https://news.stanford.edu/__data/assets/image/0021/41808/Fake-news-and-facts-digital-concept.jpeg"
                className="home-image"/>
            <h1 className="home-text10">Fake News Detection</h1>
            <span className="home-text11">User Manual</span>
            <div className="home-input">
                    <span className="home-text12">
                        Paste the news into the text field 📰
                    </span>
                <textarea
                    placeholder="Title of Your news .."
                    className="home-textarea1 textarea">
                    </textarea>
                <textarea
                    placeholder="Your news .."
                    className="home-textarea2 textarea">
                    </textarea>
                <span className="home-text13">
                        <span className="home-text14">
                          * Oops! Please add a title to your news story before proceeding.
                        </span>
                        <br></br>
                    </span>
                <span className="home-text16">
                    <span className="home-text17">
                        * Oops! The News is empty. Please add some text before proceeding.
                    </span>
                    <br></br>
                </span>
                <span className="home-text19">
            <span className="home-text20">
              * Text is too short. Please provide at least 100 characters.
            </span>
            <br></br>
          </span>
                <span className="home-text22">
            <span className="home-text23">
              * Text is too long. Please limit your input to 10,000 characters.
            </span>
            <br></br>
          </span>
                <span className="home-text25">
            <span className="home-text26">
              * Invalid input detected. Please use readable text (letters,
              numbers, and punctuation).
            </span>
            <br></br>
          </span>
                <div className="home-container2">
                    <button type="button" className="home-button1 button">
                        Reset
                    </button>
                    <button type="button" className="home-button2 button">
                        Train
                    </button>
                    <button type="button" className="home-button3 button">
                        Predict
                    </button>
                </div>
            </div>
            <svg width="24" height="24" viewBox="0 0 24 24" className="home-icon1">
                <circle r="0" cx="18" cy="12" fill="currentColor">
                    <animate
                        dur="1.5s"
                        begin=".67"
                        values="0;2;0;0"
                        calcMode="spline"
                        keySplines="0.2 0.2 0.4 0.8;0.2 0.2 0.4 0.8;0.2 0.2 0.4 0.8"
                        repeatCount="indefinite"
                        attributeName="r"
                    ></animate>
                </circle>
                <circle r="0" cx="12" cy="12" fill="currentColor">
                    <animate
                        dur="1.5s"
                        begin=".33"
                        values="0;2;0;0"
                        calcMode="spline"
                        keySplines="0.2 0.2 0.4 0.8;0.2 0.2 0.4 0.8;0.2 0.2 0.4 0.8"
                        repeatCount="indefinite"
                        attributeName="r"
                    ></animate>
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
            <div className="home-output-fake1">
                <span className="home-text28">FAKE</span>
                <span className="home-text29">🎯 Precision: 98% </span>
            </div>
            <div className="home-output-fake2">
                <span className="home-text30">FACT</span>
                <span className="home-text31">🎯 Precision: 99.9% </span>
            </div>
            <div className="home-user-manual">
                <h1 className="home-text32">
                    Welcome to the Fake News Detection Tool!
                </h1>
                <span className="home-text33">
            Whether you’re here to bust some fake news or accidentally train
            your own model into thinking everything is a conspiracy (please
            don&apos;t), this guide will walk you through every step. 
            Let&apos;s get started!
          </span>
                <div className="home-container3">
            <span className="home-text34">
              <span>Step 1: Input Some Text</span>
              <br className="home-text36"></br>
            </span>
                    <span className="home-text37">
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
                    <span className="home-text41">
              <span>Step 2: Pick an Action</span>
              <br className="home-text43"></br>
            </span>
                    <span className="home-text44">
              <span className="home-text45">
                A) Reset (For Mistakes): If you change your mind, click the
                Reset button. The text box will clear out, as if nothing ever
                happened.  Perfect for “Wait, this isn&apos;t fake news—it’s my
                to-do list!” moments.
              </span>
              <br></br>
            </span>
                    <span className="home-text47">
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
                    <span className="home-text50">
              <span>
                C) Predict (Time for Judgment): Click the Predict button. The AI
                will evaluate your text like a judge on a talent show. It will
                tell you whether the text is likely true or fake. You’ll also
                get a confidence percentage because even robots like to show
                their work. Example: “This text is 92% likely to be fake, and 8%
                likely to be true. Sorry, not sorry.”
              </span>
              <br></br>
            </span>
                    <span className="home-text53">
              <span>
                Congratulations!
                <span
                    dangerouslySetInnerHTML={{
                        __html: ' ',
                    }}
                />
              </span>
              <span className="home-text55">🔥</span>
              <span>
                {' '}
                  You now know how to use the Fake News Detection tool like a pro.
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
            <span className="home-text63">
          <span>Made by Chrzczone Chrząszcze Team </span>
          <span className="home-text65">❤️</span>
        </span>
        </div>
    </div>)
}

export default Home
