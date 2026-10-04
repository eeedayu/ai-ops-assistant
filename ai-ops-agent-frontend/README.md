# AI Ops Agent Frontend

React + Vite frontend for the AI Ops diagnosis Agent.

## Backend endpoints expected

- `POST /agent/chat`
  - body: `{ "message": "...", "device_id": "device-003" }`
  - response status: `running | waiting_for_approval | completed | failed`
- `POST /agent/{session_id}/approve`
- `POST /agent/{session_id}/reject`
  - body: `{ "reason": "..." }`

## Start

```bash
npm install
cp .env.example .env
npm run dev
```

The Vite development server proxies same-origin `/api` requests to
`http://127.0.0.1:8000`. This avoids browser CORS and mixed-content failures
when the frontend is opened through an HTTPS remote-development tunnel.

For a production static deployment, configure the web server to proxy `/api/*`
to the backend and remove the `/api` prefix before forwarding.
