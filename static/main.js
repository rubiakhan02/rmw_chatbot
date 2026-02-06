// ================= INTENT CONFIG =================

const leadKeywords = [
  "contact", "price", "pricing", "cost", "charge",
  "charges", "quote", "quotation", "hire",
  "project", "call", "email", "services",
  "interested", "talk", "budget", "estimate"
];


let leadShown = false;

function shouldShowLeadForm(msg) {
  const text = msg.toLowerCase();
  return leadKeywords.some(k => text.includes(k));
}


// ================= CHAT FUNCTION =================

async function sendMessage() {
  const input = document.getElementById('user-input');
  const message = input.value.trim();
  if (!message) return;

  addMessage('You', message);
  input.value = '';

  // ⭐ Intent trigger BEFORE bot response
  if (shouldShowLeadForm(message) && !leadShown) {
    setTimeout(() => {
      addMessage('Bot', "I can connect you with our team 👇");
      openLeadModal();
      leadShown = true;
    }, 500);
  }

  const typingIndicator = addMessage('Bot', '', true);

  try {
    const res = await fetch('http://127.0.0.1:80/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: message })
    });

    const data = await res.json();

    typingIndicator.remove();
    addMessage('Bot', data.answer, false, data.sources || []);

  } catch (err) {
    console.error(err);
    typingIndicator.remove();
    addMessage('Bot', 'Sorry, something went wrong.');
  }
}


// ================= MESSAGE UI =================

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
      sourceDiv.textContent = `Source: ${sources.join(' | ')}`;
      msg.appendChild(sourceDiv);
    }
  }

  chatBox.appendChild(msg);
  chatBox.scrollTop = chatBox.scrollHeight;
  return msg;
}


// ================= LEAD MODAL CONTROL =================

function openLeadModal() {
  document.getElementById("leadModal").style.display = "flex";
}

function closeLeadModal() {
  document.getElementById("leadModal").style.display = "none";
}


// ================= VALIDATION =================

function validateLead() {

  const name = document.getElementById("leadName").value.trim();
  const phone = document.getElementById("leadPhone").value.trim();
  const email = document.getElementById("leadEmail").value.trim();

  if (name.length < 3 || !/^[a-zA-Z ]+$/.test(name))
    return "Name must have at least 3 letters";

  if (!/^\d{10,15}$/.test(phone))
    return "Phone must be 10-15 digits";

  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email))
    return "Invalid email format";

  return null;
}


// ================= SUBMIT =================

async function submitLead() {

  const errorBox = document.getElementById("leadError");
  const error = validateLead();

  if (error) {
    errorBox.innerText = error;
    return;
  }

  errorBox.innerText = "";

  const payload = {
    name: document.getElementById("leadName").value,
    phone: document.getElementById("leadPhone").value,
    email: document.getElementById("leadEmail").value,
    message: document.getElementById("leadMsg").value
  };

  console.log("Lead Ready For API:", payload);

  // ⭐ Replace with backend API later
  // await fetch("API_URL",{...})

  closeLeadModal();
  addMessage("Bot", "Thanks! Our team will reach out soon 🙂");
}


// ================= ENTER KEY =================

document.getElementById('user-input').addEventListener('keypress', function (e) {
  if (e.key === 'Enter') sendMessage();
});  