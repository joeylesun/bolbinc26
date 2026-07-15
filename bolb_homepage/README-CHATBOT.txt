BOLB HOMEPAGE CHATBOT UPDATE
============================

Files changed
-------------
- index.html
  Adds the floating Bolb AI Assistant launcher, accessible chat panel,
  suggested questions, message area, input form, and draft disclaimer.

- assets/style.css
  Adds a responsive light-theme chatbot design using the homepage coral,
  violet, green, white, and warm-neutral palette.

- assets/site.js
  Adds open/close behavior, Escape-key handling, suggested prompts,
  typing animation, local intent matching, and draft product/application
  responses.

Current behavior
----------------
This is a front-end prototype. The assistant answers from a local JavaScript
knowledge base covering:
- UV-C LEDs
- UV-C arrays
- Reference designs
- Guardian Vision
- Air, water, surface, and food-safety applications
- Performance validation, compliance, samples, and contact inquiries

Production AI integration
-------------------------
For a true generative AI chatbot, replace generateChatReply() in
assets/site.js with a request to a server-side endpoint. The server should:
1. Store the API key securely.
2. Retrieve approved product/application content.
3. Send the grounded context and user question to the model.
4. Return the answer to the browser.

Never store an AI-provider API key directly in index.html or assets/site.js.

Content notice
--------------
Product specifications, performance claims, certifications, and application
recommendations remain draft material and should be verified before release.
