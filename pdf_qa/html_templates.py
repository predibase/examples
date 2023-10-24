css = """
<style>
.chat-message {
    padding: 1.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.bot {
    background-color: #EAF0F5
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #0E0C0C;
}
.chat-message .message-user {
  width: 80%;
  padding: 0 1.5rem;
  color: black;
}
"""

bot_template = """
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://app.predibase.com/logos/predibase/predibase.svg">
    </div>
    <div class="message">{{MSG}}</div>
</div>
"""
