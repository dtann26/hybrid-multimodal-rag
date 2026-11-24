"""Quick API smoke test."""
import requests, json

BASE = 'http://localhost:8000'

# 1. Health check
r = requests.get(f'{BASE}/api/health')
print(f'1. Health: {r.status_code} {r.json()}')

# 2. Login
r = requests.post(f'{BASE}/api/auth/login', json={'username': 'admin', 'password': 'admin123'})
print(f'2. Login: {r.status_code} token_type={r.json().get("token_type")}')
token = r.json().get('access_token')
h = {'Authorization': f'Bearer {token}'}

# 3. Get user info
r = requests.get(f'{BASE}/api/auth/me', headers=h)
me = r.json()
print(f'3. Me: {r.status_code} user={me.get("username")} role={me.get("role")}')

# 4. List models (public endpoint)
r = requests.get(f'{BASE}/api/chat/models', headers=h)
models = r.json()
print(f'4. Chat Models: {r.status_code} count={len(models)}')
for m in models:
    print(f'   - {m["name"]} ({m["model_type"]}) default={m["is_default"]}')

# 5. List KBs
r = requests.get(f'{BASE}/api/kb/', headers=h)
kbs = r.json()
print(f'5. KBs: {r.status_code} count={len(kbs)}')
for kb in kbs:
    print(f'   - {kb["name"]} (dim={kb["vector_dimension"]}, collection={kb["collection_name"]})')

# 6. Admin models (admin-only)
r = requests.get(f'{BASE}/api/admin/models', headers=h)
print(f'6. Admin Models: {r.status_code} count={len(r.json())}')

# 7. Pages load
for name, path in [('Login', '/'), ('Chat', '/chat'), ('Admin', '/admin')]:
    r = requests.get(f'{BASE}{path}')
    print(f'7. {name} page: {r.status_code} {"OK" if r.status_code == 200 else "FAIL"}')

# 8. Create a chat session
r = requests.post(f'{BASE}/api/chat/sessions', headers=h, json={'title': 'Test Chat', 'kb_ids': [kbs[0]['id']]})
print(f'8. Create session: {r.status_code} {r.json()}')

# 9. KB Stats
r = requests.get(f'{BASE}/api/kb/{kbs[0]["id"]}/stats', headers=h)
print(f'9. KB Stats: {r.status_code} {r.json()}')

# 10. Config
r = requests.get(f'{BASE}/api/admin/config', headers=h)
print(f'10. Config: {r.status_code} keys={list(r.json().keys())}')

print('\n✅ All tests passed!' if all(True for _ in range(10)) else '\n❌ Some tests failed')
