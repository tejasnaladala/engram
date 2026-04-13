import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import CinematicDemo from './CinematicDemo'
import './styles/global.css'

// Use ?demo query param or /demo hash to show cinematic visualization
const isCinematic = window.location.search.includes('demo') || window.location.hash.includes('demo')

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    {isCinematic ? <CinematicDemo /> : <App />}
  </React.StrictMode>,
)
