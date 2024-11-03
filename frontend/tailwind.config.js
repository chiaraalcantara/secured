/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}", // Ensure this includes your TSX files
  ],
  theme: {
    extend: {
      // Setting colors for theme of application
      colors: {
        primary: '#465FF1',
        secondary: '#ECF0FF',
      },
      fontFamily: {
        inter: ['Inter', 'sans-serif'], // Custom font ([primary font, fallback font])
      },
    },
  },
  plugins: [],
}