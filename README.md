# Timetable AI

Timetable AI is an AI-powered timetable generation platform that helps students create clash-free academic timetables from pre-enlistment or course selection data. The system scans pasted timetable text, detects courses, sections, faculty, and timings, then generates optimized timetable options based on student preferences.

The project is split into two repositories:

- Frontend: https://github.com/Rohithsaravanan26/timetable-frontend
- Backend: https://github.com/Rohithsaravanan26/timetable-backend

Live Website: https://www.timetable-ai.me/

## About the Project

Choosing courses during pre-enlistment can be confusing when multiple sections, faculty members, and time slots are involved. Students often need to manually compare course timings to avoid clashes and build a suitable weekly schedule.

Timetable AI solves this problem by allowing students to paste their timetable or pre-enlistment data into the platform. The application scans the text, extracts course details, detects available sections, and generates AI-ranked timetable options.

The platform is designed to help students save time, avoid timetable conflicts, and make better course selection decisions.

## Key Features

- AI-powered timetable generation
- Clash-free timetable creation
- Course and section detection from pasted text
- Faculty and timing extraction
- Student preference-based generation
- AI-ranked timetable options
- Course selection before generation
- Conflict highlighting
- PDF download support
- Image export support
- Favourite timetable saving
- Google sign-in support
- Trial and credit-based usage flow
- Faculty review submission
- Faculty insight viewing
- Responsive web interface
- Separate frontend and backend architecture

## Live Platform Features

The live application supports:

- Google account sign-in
- Timetable text input
- Course scanning
- Course selection
- Preference-based timetable generation
- AI-ranked results
- Conflict marking
- PDF export
- Image export
- Favourite timetable saving
- Faculty reviews
- Credit-based generation system

## Tech Stack

### Frontend

- React / Next.js
- JavaScript / TypeScript
- CSS / Tailwind CSS
- Vercel deployment

### Backend

- Node.js
- API-based timetable processing
- AI logic for timetable generation
- Authentication and credit-related backend support

### AI / Logic

- Timetable parsing
- Course extraction
- Section detection
- Clash detection
- Preference-based ranking
- AI-assisted schedule optimization

### Deployment

- Frontend deployed on Vercel
- Backend deployed separately
- Custom domain connected to the frontend

## Project Architecture

```text
Timetable AI/
│
├── timetable-frontend/
│   ├── app/ or src/
│   ├── components/
│   ├── public/
│   ├── styles/
│   ├── package.json
│   └── README.md
│
└── timetable-backend/
    ├── routes/
    ├── controllers/
    ├── services/
    ├── utils/
    ├── package.json
    └── README.md
```

## How It Works

1. Student opens the Timetable AI website.
2. Student signs in using Google.
3. Student pastes pre-enlistment or timetable data.
4. The system scans the pasted text.
5. Courses, sections, timings, and faculty details are detected.
6. Student selects the courses they want to enroll in.
7. Student sets preferences such as avoiding specific time slots or preferring weekends off.
8. The AI engine generates timetable options.
9. Results are ranked based on clashes, preferences, and schedule quality.
10. Student can download the timetable as PDF or export it as an image.
11. Student can save favourite timetable options.

## Preference Options

The platform can support preferences such as:

- Avoid 8–10 AM slots
- Avoid 10–12 AM slots
- Avoid 1–3 PM slots
- Avoid 3–5 PM slots
- Prefer weekend off
- Preferred faculty
- Avoid faculty
- Clash-free scheduling
- Better time distribution

## Frontend Repository

Repository:

```text
https://github.com/Rohithsaravanan26/timetable-frontend
```

The frontend handles:

- User interface
- Timetable input
- Course scanning display
- Course selection
- Preference selection
- Result visualization
- PDF/image export
- Favourite timetable UI
- Faculty review interface
- Authentication UI
- Credit/trial display

## Backend Repository

Repository:

```text
https://github.com/Rohithsaravanan26/timetable-backend
```

The backend handles:

- Timetable parsing
- Course extraction
- Section processing
- Clash detection
- Timetable generation logic
- AI ranking
- User-related backend operations
- Credit/trial logic
- API responses for frontend

## Installation and Setup

Clone both repositories.

### 1. Clone Frontend

```bash
git clone https://github.com/Rohithsaravanan26/timetable-frontend.git
```

```bash
cd timetable-frontend
```

```bash
npm install
```

### 2. Clone Backend

Open a new terminal.

```bash
git clone https://github.com/Rohithsaravanan26/timetable-backend.git
```

```bash
cd timetable-backend
```

```bash
npm install
```

## Environment Variables

Create environment files based on your project setup.

### Frontend `.env.local`

```env
NEXT_PUBLIC_API_URL=your_backend_api_url
NEXT_PUBLIC_GOOGLE_CLIENT_ID=your_google_client_id
NEXT_PUBLIC_SITE_URL=https://www.timetable-ai.me
```

### Backend `.env`

```env
PORT=5000
FRONTEND_URL=http://localhost:3000
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
JWT_SECRET=your_jwt_secret
AI_API_KEY=your_ai_api_key
```

Update these values based on the actual services used in your project.

Do not commit `.env` or `.env.local` files to GitHub.

## Running the Project Locally

### Start Backend

```bash
cd timetable-backend
npm run dev
```

Backend local URL example:

```text
http://localhost:5000
```

### Start Frontend

Open another terminal.

```bash
cd timetable-frontend
npm run dev
```

Frontend local URL example:

```text
http://localhost:3000
```

## Available Scripts

### Frontend

```bash
npm run dev
```

Runs the frontend in development mode.

```bash
npm run build
```

Builds the frontend for production.

```bash
npm start
```

Starts the production frontend server.

```bash
npm run lint
```

Runs lint checks if configured.

### Backend

```bash
npm run dev
```

Runs the backend in development mode.

```bash
npm start
```

Starts the backend server.

## API Overview

The backend can include API routes for:

```text
POST /scan
POST /generate
POST /favourites
GET /favourites
POST /reviews
GET /faculty-insights
POST /credits
GET /user
```

Actual endpoint names may differ based on implementation.

## Deployment

### Frontend Deployment

The frontend is deployed at:

```text
https://www.timetable-ai.me/
```

Recommended deployment platform:

```text
Vercel
```

Steps:

1. Push frontend code to GitHub.
2. Import the frontend repository into Vercel.
3. Add frontend environment variables.
4. Connect the custom domain.
5. Deploy.

### Backend Deployment

The backend can be deployed using platforms such as:

- Render
- Railway
- Vercel Serverless Functions
- Fly.io
- Any Node.js hosting provider

Steps:

1. Push backend code to GitHub.
2. Deploy backend to a Node.js-compatible platform.
3. Add backend environment variables.
4. Copy backend production URL.
5. Add it to the frontend environment variable.

## Screenshots

Add screenshots inside a `docs` or `public/screenshots` folder and update the paths below.

```markdown
![Landing Page](docs/landing-page.png)
![Timetable Input](docs/timetable-input.png)
![Course Detection](docs/course-detection.png)
![Generated Timetable](docs/generated-timetable.png)
![Faculty Reviews](docs/faculty-reviews.png)
```

## Use Cases

Timetable AI can be used for:

- Student course selection
- Pre-enlistment planning
- Clash-free timetable generation
- Faculty-based course preference planning
- Weekly academic schedule planning
- Fast timetable comparison
- Exporting timetable as PDF or image
- Saving favourite timetable combinations

## Future Improvements

- Add more accurate timetable parsing
- Add support for more timetable formats
- Add department-wise course templates
- Add semester-wise course grouping
- Add drag-and-drop timetable editing
- Add automatic clash explanation
- Add multiple timetable comparison view
- Add calendar sync
- Add mobile app version
- Add admin dashboard
- Add faculty rating analytics
- Add timetable sharing links
- Add smarter AI ranking
- Add offline timetable export
- Add user profile and history
- Add payment gateway integration for credits
- Add institution-level timetable management

## Project Status

Timetable AI is currently under development. The live version includes timetable scanning, course detection, preference-based generation, AI-ranked results, export options, favourites, sign-in, credits, and faculty review-related features.

## Author

Rohith Saravanan

GitHub: https://github.com/Rohithsaravanan26

## Acknowledgement

This project was created to help students generate better academic timetables with less manual effort. Timetable AI aims to reduce course selection confusion by combining timetable parsing, clash detection, student preferences, and AI-assisted schedule generation.

## License

This project can be released under the MIT License.

You can add a `LICENSE` file to the repository if you want others to use, modify, and contribute to the project.

Recommended license:

```text
MIT License
```
