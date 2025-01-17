import React from 'react'
import {BrowserRouter, Route, Routes} from 'react-router-dom';
// Relative imports
import './style.css'
import Home from './views/home/home.js';
import NotFound from './views/not-found/not-found.js';

const App = () => {
    return (
        <BrowserRouter>
            <Routes>
                <Route path="/" element={<Home/>}/>
                {/*<Route path="/404" element={<NotFound/>}/>*/}
                <Route path="*" element={<NotFound/>}/>
            </Routes>
        </BrowserRouter>
    )
}

export default App;