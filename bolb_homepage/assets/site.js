(function(){
  const toggles = document.querySelectorAll('[data-menu-toggle]');
  const menu = document.querySelector('[data-menu]');
  function setMenu(open){
    if(!menu) return;
    menu.classList.toggle('open',open);
    document.body.classList.toggle('menu-open',open);
    toggles.forEach(t=>t.setAttribute('aria-expanded',String(open)));
  }
  toggles.forEach(t=>t.addEventListener('click',()=>setMenu(!menu.classList.contains('open'))));
  if(menu){
    menu.addEventListener('click',e=>{if(e.target.closest('a')) setMenu(false)});
    document.addEventListener('click',e=>{if(menu.classList.contains('open')&&!menu.contains(e.target)&&![...toggles].some(t=>t.contains(e.target))) setMenu(false)});
    document.addEventListener('keydown',e=>{if(e.key==='Escape') setMenu(false)});
  }
  const els=document.querySelectorAll('.reveal');
  if('IntersectionObserver' in window){
    const io=new IntersectionObserver(entries=>entries.forEach(entry=>{if(entry.isIntersecting){entry.target.classList.add('in');io.unobserve(entry.target)}}),{threshold:.12,rootMargin:'0px 0px -25px'});
    els.forEach(el=>io.observe(el));
  }else els.forEach(el=>el.classList.add('in'));
  document.querySelectorAll('[data-year]').forEach(el=>el.textContent=new Date().getFullYear());


  /* Bolb AI Assistant: local draft knowledge base.
     Replace generateChatReply() with a server-side API request for a production AI assistant.
     Never place an API key in this browser file. */
  const chatToggle=document.getElementById('chatToggle');
  const chatPanel=document.getElementById('chatPanel');
  const chatClose=document.getElementById('chatClose');
  const chatBody=document.getElementById('chatBody');
  const chatForm=document.getElementById('chatForm');
  const chatInput=document.getElementById('chatInput');
  const chatSuggestions=document.getElementById('chatSuggestions');

  if(chatToggle&&chatPanel&&chatBody&&chatForm&&chatInput){
    function setChat(open){
      chatPanel.classList.toggle('open',open);
      chatPanel.setAttribute('aria-hidden',String(!open));
      chatToggle.setAttribute('aria-expanded',String(open));
      chatToggle.setAttribute('aria-label',open?'Close Bolb AI Assistant':'Open Bolb AI Assistant');
      if(open){setMenu(false);window.setTimeout(()=>chatInput.focus(),120)}
    }
    chatToggle.addEventListener('click',()=>setChat(!chatPanel.classList.contains('open')));
    if(chatClose) chatClose.addEventListener('click',()=>setChat(false));
    document.addEventListener('keydown',e=>{if(e.key==='Escape'&&chatPanel.classList.contains('open'))setChat(false)});

    const CHAT_KB=[
      {keys:['which product','start with','choose product','recommend product','product should'],reply:'Start with the system architecture. Choose UV-C LEDs when you want maximum control over emitter placement, optics, drive, and thermal design. Choose UV-C arrays when you need scalable optical power or a defined irradiation pattern. Use a reference design to accelerate an early prototype. Guardian Vision is the safety-and-control option for selected applications.\n\nProduct pages: UV-C LEDs, UV-C arrays, reference designs, and Guardian Vision are available from the Products section.'},
      {keys:['led','emitter','s3535','s6060','wavelength','265','275'],reply:'Bolb’s UV-C LED page presents discrete emitter formats for teams designing their own optical, electrical, and thermal architecture. Key selection variables include peak wavelength, optical output at the intended drive condition, package geometry, viewing angle, thermal resistance, and lifetime assumptions. Open “UV-C LEDs” under Products for the draft selection table.'},
      {keys:['array','arrays','multi-emitter','multi emitter'],reply:'UV-C arrays combine multiple emitters to increase optical power and shape irradiance across an air channel, water reactor, surface zone, or process line. Array selection should be coordinated with board geometry, current distribution, optics, cooling, and the required dose uniformity. Open “UV-C arrays” under Products for candidate formats.'},
      {keys:['reference','prototype','integration','design'],reply:'Reference designs are intended to shorten the path from an application requirement to a working prototype. The draft page organizes source selection, driver and thermal concepts, mechanical integration, measurement, and validation considerations. They are starting points—not substitutes for application-specific verification.'},
      {keys:['guardian','vision','camera','presence','safety','human','control'],reply:'Guardian Vision is positioned as one product within the wider Bolb portfolio. It supports selected UV-C systems that require presence awareness, operating logic, and coordinated cycle control. Its role is system safety and control rather than the core LED platform. Open the Guardian Vision product page for the current draft story.'},
      {keys:['air','hvac','duct','airflow'],reply:'Air systems must deliver sufficient UV-C dose while air is moving through the optical zone. Important variables include airflow, residence time, irradiance distribution, pressure drop, fouling, thermal conditions, and access for service. The Air application page explains the draft system story and product parameters.'},
      {keys:['water','flow','reactor','uvt','transmittance'],reply:'Water systems are governed by flow rate, UV transmittance, absorption, optical path length, mixing, reactor geometry, and surface fouling. The source and reactor must be designed together so every pass receives an appropriate dose. Open the Water application page for the draft workflow and parameter list.'},
      {keys:['surface','surfaces','shadow','high-touch','high touch'],reply:'Surface treatment depends on distance, angle, shadows, material reflectance, target area, exposure time, and human-access controls. Arrays or linear emitter layouts can help improve coverage, but the complete treatment zone should be measured and validated. Open the Surfaces application page for details.'},
      {keys:['food','produce','packaging','processing','shelf'],reply:'Food-safety applications may place UV-C in processing, packaging, storage, or conveyance steps. Product geometry, line speed, dose uniformity, shadowing, temperature, moisture, and material compatibility all matter. Microbial and product-quality validation should be performed under representative conditions.'},
      {keys:['compare applications','air and water','applications different'],reply:'Air is mainly a moving-gas and residence-time problem; water adds absorption, optical path length, and reactor mixing; surfaces are dominated by geometry and shadowing; food safety combines surface or fluid treatment with line speed and product-quality constraints. Each application page connects these operating conditions to candidate LED or array parameters.'},
      {keys:['performance','kill','reduction','99.99','99.96','dose'],reply:'The homepage includes draft performance summaries derived from earlier Bolb materials. Actual reduction depends on organism, dose, geometry, flow, environmental conditions, and test method. Please confirm all performance claims with current Bolb validation reports before using them in a specification or public statement.'},
      {keys:['certification','ul','rohs','reach','ozone','mercury'],reply:'The site describes a mercury-free solid-state UV-C platform and includes draft references to compliance information. Exact certifications, scope, model applicability, and current documentation should be confirmed with Bolb before publication or procurement.'},
      {keys:['price','quote','sample','buy','purchase','contact','sales','expert'],reply:'For pricing, samples, datasheets, or an application review, email info@bolb.co. It helps to include your target application, required treatment rate or geometry, expected volume, operating environment, and project timing.'},
      {keys:['blog','article','insight'],reply:'The homepage keeps a Blog and Insights section covering UV-C science, air applications, and water-system design. In this draft, the cards currently route to related application stories; they can later be connected to a CMS-backed blog.'},
      {keys:['hello','hi','hey'],reply:'Hello! Ask me about UV-C LEDs, arrays, reference designs, Guardian Vision, or the Air, Water, Surfaces, and Food Safety application paths.'}
    ];

    function generateChatReply(question){
      const normalized=question.toLowerCase();
      let best=null,bestScore=0;
      CHAT_KB.forEach(item=>{
        const score=item.keys.reduce((sum,key)=>sum+(normalized.includes(key)?1:0),0);
        if(score>bestScore){best=item;bestScore=score}
      });
      return best?best.reply:'I do not have a reliable draft answer for that yet. Please try asking about a product family, an application, Guardian Vision, performance validation, or contact info@bolb.co for an engineering review.';
    }
    function addChatMessage(role,text){
      const message=document.createElement('div');
      message.className='chat-message '+role;
      message.textContent=text;
      chatBody.appendChild(message);
      chatBody.scrollTop=chatBody.scrollHeight;
    }
    function showChatTyping(){
      const typing=document.createElement('div');
      typing.className='chat-typing';typing.id='chatTyping';
      typing.innerHTML='<span></span><span></span><span></span>';
      chatBody.appendChild(typing);chatBody.scrollTop=chatBody.scrollHeight;
    }
    function hideChatTyping(){const typing=document.getElementById('chatTyping');if(typing)typing.remove()}
    function submitChat(value){
      const question=(value||'').trim();if(!question)return;
      addChatMessage('user',question);chatInput.value='';
      if(chatSuggestions)chatSuggestions.hidden=true;
      showChatTyping();
      window.setTimeout(()=>{hideChatTyping();addChatMessage('bot',generateChatReply(question))},550+Math.random()*350);
    }
    chatForm.addEventListener('submit',e=>{e.preventDefault();submitChat(chatInput.value)});
    if(chatSuggestions)chatSuggestions.addEventListener('click',e=>{const button=e.target.closest('[data-question]');if(button)submitChat(button.dataset.question)});
  }

})();
