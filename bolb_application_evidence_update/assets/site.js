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
      {keys:['which product','start with','choose product','recommend','component'],reply:'Bolb sells packaged UV-C LEDs, modules, arrays, and engineering support. Start with wavelength, required irradiance or dose, working distance, geometry, thermal constraints, and production volume. Open Products to compare S3535-H, S3535-F, S6060-TL, Hex, 1×12, 5×5, and custom formats.'},
      {keys:['s3535-h','s3535 h','150 mw','350 ma'],reply:'S3535-H is Bolb’s higher-output 3.5 × 3.5 mm 265 nm platform. The public catalog summarizes typical 150 mW optical output at 350 mA, but current bin, forward voltage, lifetime, and availability must be confirmed with the controlled datasheet.'},
      {keys:['s3535-f','s3535 f','100 ma','flat window'],reply:'S3535-F is a 3.5 × 3.5 mm flat-window 265 nm package intended for lower-current or distributed layouts. The public catalog summarizes 35–45 mW at 100 mA for a stated bin; confirm the exact production bin and datasheet.'},
      {keys:['s6060','275','focused','quartz lens'],reply:'S6060-TL is a 6 × 6 mm 275 nm option with a focused-lens configuration. The public catalog summarizes roughly 90 mW at 250 mA. Confirm the exact variant, beam, lifetime, and availability before designing it in.'},
      {keys:['array','module','hex','1x12','1×12','5x5','5×5'],reply:'Bolb module formats include S3535-H Hex, a linear 1×12 array, a dense 5×5 array, and custom arrays. Choose from treatment-field shape, required output, driver architecture, cooling, connector, serviceability, and production requirements.'},
      {keys:['air','upper room','upper-room','targeted','hvac'],reply:'For air systems, Bolb supplies the UV-C source—not the finished fixture. The Air page separates component options from customer-system evidence: upper-room and targeted reference implementations show what complete systems can achieve under stated room, airflow, organism, and exposure conditions.'},
      {keys:['water','reactor','flow','uvt','transmittance','b-w10t','bw10t'],reply:'Water light-engine selection depends on flow, UV transmittance, optical path, mixing, target dose, fouling, and cooling. The B-W10T page example reports a ≥99.9% E. coli target at 10 T/H for that reference reactor; it is a system result under its test conditions, not a universal LED-only rate.'},
      {keys:['surface','hospital','warehouse','shopping mall','suvos','yolo','multi-zone'],reply:'Surface treatment may use compact packaged-LED fixtures or multiple independently controlled fixtures. The compact customer sheet contains calculated organism-specific examples; SUVOS demonstrates occupancy-aware zone control but has no standalone microbial-reduction claim. Bolb’s commercial role is the LED/module platform and integration support.'},
      {keys:['food','conveyor','packaging','processing'],reply:'Food applications may use S3535-F for distributed emitters, S3535-H for higher-output points, 1×12 arrays for linear fields, or custom arrays for equipment geometry. Validate microbial reduction and product quality under the actual process conditions.'},
      {keys:['evidence','kill rate','reduction rate','99.9','efficacy','deactivation'],reply:'Microbial reduction is a complete-system result, not a property of an LED alone. We label each website example as tested, customer-reported, or calculated and include the organism, dose or exposure time, flow or room geometry, and claim boundary.'},
      {keys:['customer','powered by bolb','finished system','fixture'],reply:'“Powered by Bolb” means the finished system was developed by a customer or partner using Bolb UV-C technology. Bolb supplies LEDs, modules, arrays, and engineering support; the system manufacturer owns final specifications, certification, warranty, and support.'},
      {keys:['reference','custom','prototype','engineering'],reply:'Bolb reference-design support can cover source selection, optical layout, driver and current regulation, thermal interfaces, safety concepts, dose mapping, and prototype planning. The OEM remains responsible for the complete system and validation.'},
      {keys:['price','quote','sample','buy','purchase','contact','sales'],reply:'Email info@bolb.co with the application, wavelength, output or dose target, geometry, flow or line speed, cooling constraints, prototype timing, expected volume, and the LED or module family of interest.'},
      {keys:['hello','hi','hey'],reply:'Hello! Bolb supplies UV-C LEDs and modules. Tell me your application and optical, geometric, thermal, or flow requirements, and I’ll guide you to a component family.'}
    ];

    function generateChatReply(question){
      const normalized=question.toLowerCase();
      let best=null,bestScore=0;
      CHAT_KB.forEach(item=>{
        const score=item.keys.reduce((sum,key)=>sum+(normalized.includes(key)?1:0),0);
        if(score>bestScore){best=item;bestScore=score}
      });
      return best?best.reply:'I do not have a reliable draft answer for that yet. Please try asking about Air, Water, Healthcare Surfaces, Food Safety, performance validation, or contact info@bolb.co for an engineering review.';
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
