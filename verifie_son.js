/*  Le rejeu joué hors navigateur.

    Le son du rejeu ne se laisse pas vérifier depuis Python : il vit dans le
    navigateur. On remplace donc le DOM et la carte son par des doublures, on
    exécute le lecteur tel qu'il sera livré, on clique sur ses boutons et on
    compte ce qui sonne.

        node verifie_son.js chemin/vers/lecteur.html

    Appelé automatiquement par verifie.py quand Node est présent. */
const fs = require('fs');
const html = fs.readFileSync(process.argv[2], 'utf8');
const src = html.match(/<script>([\s\S]*?)<\/script>/)[1];

let oscillateurs = 0, gains = 0, delais = 0, reprises = 0;
const clics = {}, els = {};

function El(id){
  const cls = new Set();
  return {id, style:{setProperty(){}}, textContent:'', innerHTML:'',
    classList:{ add:c=>cls.add(c), remove:c=>cls.delete(c),
      contains:c=>cls.has(c), toggle:(c,f)=>{ const v = f===undefined ? !cls.has(c) : f;
        v?cls.add(c):cls.delete(c); return v; } },
    _cls: cls,
    addEventListener:(t,f)=>{ if(t==='click') clics[id]=f; },
    appendChild(){}, removeChild(){}, querySelector:()=>null, querySelectorAll:()=>[],
    setAttribute(){}, getAttribute:()=>null, parentNode:null,
    currentTime: 3.0, duration: 12, paused: false, requestFullscreen:null};
}
function get(id){ return els[id] || (els[id] = El(id)); }

global.document = { getElementById:get, createElement:()=>El('neuf'),
  addEventListener(){}, fullscreenElement:null, exitFullscreen(){} };
global.performance = { now:()=>0 };
global.requestAnimationFrame = ()=>0;
global.cancelAnimationFrame = ()=>0;
let boucle = null;
global.setInterval = (f)=>{ boucle = f; return 1; };
global.clearInterval = ()=>{};
global.setTimeout = (f)=>{ f(); return 1; };

function Param(){ return { value:0, setValueAtTime(){}, linearRampToValueAtTime(){},
  exponentialRampToValueAtTime(){}, cancelScheduledValues(){} }; }
function Noeud(extra){ return Object.assign({ connect(){}, disconnect(){} }, extra||{}); }
global.AudioContext = function(){
  this.currentTime = 0; this.state = 'running';
  this.resume = ()=>{ reprises++; };
  this.destination = Noeud();
  this.createGain = ()=>{ gains++; return Noeud({gain:Param()}); };
  this.createOscillator = ()=>{ oscillateurs++;
    return Noeud({type:'', frequency:Param(), start(){}, stop(){}}); };
  this.createBiquadFilter = ()=>Noeud({type:'', frequency:Param(), Q:Param()});
  this.createDelay = ()=>{ delais++; return Noeud({delayTime:Param()}); };
};
global.window = { AudioContext: global.AudioContext };

// ── on exécute le lecteur ──────────────────────────────────────────────
new Function(src)();

let ok = 0, ko = 0;
function t(nom, fn){ try{ fn(); console.log('  ✔ ' + nom); ok++; }
  catch(e){ console.log('  ✘ ' + nom + ' → ' + e.message); ko++; } }
function vrai(c, m){ if(!c) throw new Error(m); }

console.log('── les boutons sont bien distincts ──');
t('les cinq boutons ont chacun leur gestionnaire', ()=>{
  ['kxson','kxair','kxleg','kxmet','kxplein'].forEach(id =>
    vrai(typeof clics[id] === 'function', id + ' sans gestionnaire'));
});

console.log('── le son ──');
t('rien ne sonne avant le clic', ()=>{
  vrai(oscillateurs === 0, oscillateurs + ' sons joués sans rien demander');
});
t('le clic fait sonner un accord immédiatement', ()=>{
  clics['kxson']();
  vrai(oscillateurs > 0, "rien ne sonne à l'instant du clic");
});
t('la musique se déroule ensuite', ()=>{
  vrai(typeof boucle === 'function', 'aucune horloge de lecture');
  //  on déroule les douze secondes de vidéo, 60 ms par tour
  const ctx = els['kxson'];  // repère inutile, juste pour la lisibilité
  for(let x = 0; x < 12; x += 0.06){
    get('kxv').currentTime = x;
    boucle();
  }
  vrai(oscillateurs > 12, 'trop peu de sons sur douze secondes : ' + oscillateurs);
  vrai(delais > 0, "pas d'écho");
});
t('la musique joue même si la vidéo refuse de démarrer', ()=>{
  //  on repart de zéro, vidéo arrêtée : c'est le cas des lecteurs qui
  //  bloquent la lecture automatique dans un cadre incorporé
  const avant = oscillateurs;
  clics['kxson']();                        // extinction
  get('kxv').paused = true;
  clics['kxson']();                        // rallumage, vidéo à l'arrêt
  for(let i = 0; i < 220; i++) boucle();   // ~13 secondes d'horloge propre
  vrai(oscillateurs > avant + 12,
       'muet quand la vidéo est arrêtée (' + (oscillateurs - avant) + ' sons)');
  get('kxv').paused = false;
});
t("♪ son n'ouvre pas le panneau des mesures", ()=>{
  vrai(!get('kxpan')._cls.has('ouvert'), 'le panneau des mesures s’est ouvert');
  vrai(!get('kxlegp')._cls.has('ouvert'), 'la légende s’est ouverte');
});
t("le bouton s'allume et annonce l'œuvre", ()=>{
  vrai(get('kxson')._cls.has('actif'), 'le bouton ne se marque pas actif');
  vrai(/^♪ .{3,}/.test(get('kxson').textContent),
       'libellé inattendu : ' + get('kxson').textContent);
  vrai(/(Bach|Beethoven|Pachelbel)/.test(get('kxson').title || ''),
       "l'infobulle ne nomme pas l'auteur : " + get('kxson').title);
});
t('⟳ change de morceau sans couper le son', ()=>{
  const titre = get('kxson').textContent, avant = oscillateurs;
  clics['kxair']();
  vrai(get('kxson').textContent !== titre,
       'le morceau ne change pas (' + titre + ')');
  vrai(oscillateurs > avant, 'le nouveau morceau ne sonne pas');
  vrai(get('kxson')._cls.has('actif'), "l'écoute a été coupée");
});
t('⟳ fait le tour des cinq œuvres et revient', ()=>{
  const vus = new Set([get('kxson').textContent]);
  for(let i = 0; i < 4; i++){ clics['kxair'](); vus.add(get('kxson').textContent); }
  vrai(vus.size === 5, 'seulement ' + vus.size + ' morceaux distincts');
  clics['kxair']();
  vrai(vus.has(get('kxson').textContent), 'le tour ne revient pas au début');
});
t('un second clic coupe le son', ()=>{
  const avant = oscillateurs;
  clics['kxson']();
  vrai(!get('kxson')._cls.has('actif'), 'le bouton reste actif');
  vrai(oscillateurs === avant, 'le son continue après extinction');
});

console.log('── les autres boutons ──');
t('« métriques » ouvre bien le panneau des mesures', ()=>{
  clics['kxmet']();
  vrai(get('kxpan')._cls.has('ouvert'), 'le panneau ne s’ouvre pas');
});
t('« ? » ouvre la légende et referme les mesures', ()=>{
  clics['kxleg']();
  vrai(get('kxlegp')._cls.has('ouvert'), 'la légende ne s’ouvre pas');
  vrai(!get('kxpan')._cls.has('ouvert'), 'les mesures restent ouvertes');
});

console.log('\n' + (ko ? '✘ ' + ko + ' échec(s)' : '✔ ' + ok + ' contrôles passés, 0 échec'));
console.log('   oscillateurs joués : ' + oscillateurs + ' · gains : ' + gains + ' · échos : ' + delais);
process.exit(ko ? 1 : 0);
