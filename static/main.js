async function sendMessage() {
  const input = document.getElementById('user-input');
  const message = input.value.trim();
  if (!message) return;

  addMessage('You', message);
  input.value = '';

  const typingIndicator = addMessage('Bot', '', true); // typing animation

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: message })
    });
    const data = await res.json();

    typingIndicator.remove();
    addMessage('Bot', data.answer, false, data.sources);

  } catch (err) {
    console.error(err);
    typingIndicator.remove();
    addMessage('Bot', 'Sorry, something went wrong.');
  }
}

function addMessage(sender, text, isTyping = false, sources = []) {
  const chatBox = document.getElementById('chat-box');
  const msg = document.createElement('div');
  msg.className = 'message ' + (sender === 'You' ? 'user-message' : 'bot-message');

  if (isTyping) {
    msg.innerHTML = `<div class="typing"><span></span><span></span><span></span></div>`;
  } else {
    msg.textContent = text;

    if (sources.length > 0) {
      const sourceDiv = document.createElement('div');
      sourceDiv.className = 'bot-source';
      sourceDiv.textContent = `Source: ${sources.map(s => s.text).join(' | ')}`;
      msg.appendChild(sourceDiv);
    }
  }

  chatBox.appendChild(msg);
  chatBox.scrollTop = chatBox.scrollHeight;
  return msg;
}

document.getElementById('user-input').addEventListener('keypress', function(e) {
  if (e.key === 'Enter') sendMessage();
});
