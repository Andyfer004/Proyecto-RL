/** Convert the frozen report narrative to typeset LaTeX. No experiment execution. */
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
const require=createRequire(import.meta.url);
const {marked}=require('../docs/assets/vendor/marked/marked.cjs');
const root=path.resolve('.');
const source=fs.readFileSync('docs/informe_final.md','utf8');
const equations=[];
let md=source.replace(/\$\$([\s\S]+?)\$\$|\$([^$\n]+?)\$/g,(_,block,inline)=>{
 const index=equations.length;equations.push({value:block??inline,block:block!==undefined});return `ZZFORMULA${index}ZZ`;
});
// Some legacy inline formulas use plain parentheses; restore them for typesetting.
let converted='';
for(let start=0;start<md.length;){
 if(md[start]!=='(' || md[start-1]===']' || /[A-Za-z0-9]/.test(md[start-1]??'')){converted+=md[start++];continue;}
 let end=start+1,depth=1;
 for(;end<md.length&&depth;end++){if(md[end]==='(')depth++;if(md[end]===')')depth--;if(md[end]==='\n')break;}
 const value=md.slice(start+1,end-1);
 if(depth===0&&!value.includes('`')&&value.length<500&&(value.includes('\\')||((value.includes('_')||value.includes('^'))&&!value.includes(' '))||/^[A-Za-z][A-Za-z0-9_{}/=.+-]*$/.test(value))){
  const index=equations.length;equations.push({value,block:false});converted+=`ZZFORMULA${index}ZZ`;start=end;
 }else{converted+=md[start++];}
}
md=converted;

function esc(s){
 const fragments=[];
 const hold=value=>{const key=`ZZTEXSYMBOL${fragments.length}ZZ`;fragments.push(value);return key;};
 const greek={'θ':'theta','ω':'omega','τ':'tau','Φ':'Phi','ψ':'psi','Δ':'Delta','γ':'gamma','ε':'epsilon','σ':'sigma','π':'pi','α':'alpha','β':'beta','Σ':'Sigma','δ':'delta','λ':'lambda','∈':'in','≤':'le','≥':'ge','±':'pm','→':'rightarrow','×':'times','·':'cdot','∣':'vert'};
 let value=String(s).replace(/[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+/g,text=>hold('\\textsuperscript{'+[...text].map(c=>({'⁰':'0','¹':'1','²':'2','³':'3','⁴':'4','⁵':'5','⁶':'6','⁷':'7','⁸':'8','⁹':'9','⁻':'-','⁺':'+'}[c])).join('')+'}'));
 value=value.replace(/[¿¡]/g,c=>hold(c==='¿'?'\\textquestiondown{}':'\\textexclamdown{}'));
 value=value.replace(/[θωτΦψΔγεσπαβΣδλ∈≤≥±→×·∣]/g,c=>hold('\\ensuremath{\\'+greek[c]+'}'));
 const escapes={'\\':'\\textbackslash{}','&':'\\&','%':'\\%','$':'\\$','#':'\\#','_':'\\_','{':'\\{','}':'\\}','~':'\\textasciitilde{}','^':'\\textasciicircum{}'};
 value=value.replace(/[\\&%$#_{}~^]/g,c=>escapes[c]).replace(/\//g,'/\\allowbreak{}').replace(/,/g,',\\allowbreak{}').replace(/Plataforma/g,'Plata\\-forma').replace(/[–—‑−]/g,'-');
 fragments.forEach((fragment,index)=>{value=value.split(`ZZTEXSYMBOL${index}ZZ`).join(fragment);});
 return value;
}
function plainCode(s){return '\\nolinkurl{'+s.replace(/[{}]/g,'')+'}';}
function inline(tokens=[]){return tokens.map(t=>{
 switch(t.type){
 case 'text':return t.tokens?inline(t.tokens):esc(t.text);
 case 'escape':return esc(t.text);
 case 'strong':return '\\textbf{'+inline(t.tokens)+'}';
 case 'em':return '\\emph{'+inline(t.tokens)+'}';
 case 'codespan':return plainCode(t.text);
 case 'link':return /^https?:/.test(t.href)?'\\href{'+t.href.replace(/%/g,'\\%')+'}{'+inline(t.tokens)+'}':'\\textcolor{Accent}{'+inline(t.tokens)+'}';
 case 'image':return figure(t.href,t.text);
 case 'br':return '\\\\ ';
 default:return esc(t.text??t.raw??'');
 }
 }).join('');}
function figure(href,caption){
 const mapped=href.endsWith('alcance_exitos.png')?'tmp/pdfs/exitos_pdf.pdf':href.endsWith('alcance_contacto_consumo.png')?'tmp/pdfs/contacto_pdf.pdf':null;
 const src=mapped?path.resolve(mapped):path.resolve('docs',href);
 if(!fs.existsSync(src))throw new Error('Missing figure '+src);
 return '\n\\begin{figure}[htbp]\\centering\n\\includegraphics[width=\\linewidth]{'+src+'}\n\\caption{'+esc(caption)+'}\\end{figure}\n';
}
function table(t){
 const n=t.header.length;const landscape=n>=9;const width=landscape?'.25':n>=6?'.30':n===5?'.24':'.32';
 const first=Number(width);const rest=(1-first)/(n-1);
 const col=(frac)=>`>{\\raggedright\\arraybackslash}p{${frac.toFixed(4)}\\dimexpr\\linewidth-${2*n*3}pt\\relax}`;
 const cols=col(first)+Array(n-1).fill(col(rest)).join('');
 const row=cells=>cells.map(c=>inline(c.tokens)).join(' & ')+' \\\\';
 const head='\\rowcolor{Pale}'+row(t.header)+'\n\\midrule\n';
 return `${landscape?'\\begin{landscape}\n':''}\\begingroup\n\\setlength{\\tabcolsep}{3pt}\n\\renewcommand{\\arraystretch}{1.35}\n\\fontsize{${landscape?'10.5':'9.1'}}{${landscape?'13.5':'11.8'}}\\selectfont\n\\begin{longtable}{${cols}}\n\\toprule\n${head}\\endfirsthead\n\\toprule\n${head}\\endhead\n\\bottomrule\\endfoot\n${t.rows.map(row).join('\n')}\n\\end{longtable}\n\\endgroup\n${landscape?'\\end{landscape}\n':''}`;
}
function blocks(tokens){return tokens.map(t=>{
 switch(t.type){
 case 'space':return '\n';
 case 'heading':return t.depth===1?'':`\n\\needspace{6\\baselineskip}\n\\${t.depth===2?'section':'subsection'}{${inline(t.tokens).replace(/^\d+(?:\.\d+)*\.?\s+/,'')}}\n`;
 case 'paragraph':return inline(t.tokens)+'\n\n';
 case 'table':return table(t);
 case 'list':return `\\begin{${t.ordered?'enumerate':'itemize'}}\n`+t.items.map(x=>'\\item '+blocks(x.tokens)).join('\n')+`\\end{${t.ordered?'enumerate':'itemize'}}\n`;
 case 'text':return t.tokens?inline(t.tokens):esc(t.text);
 case 'code':return '\n\\begin{CodeBlock}\n'+t.text.replace(/[–—‑]/g,'-')+'\n\\end{CodeBlock}\n';
 case 'blockquote':return '\\begin{quote}'+blocks(t.tokens)+'\\end{quote}\n';
 case 'hr':return '\\medskip\\hrule\\medskip\n';
 default:throw new Error('Unsupported block '+t.type);
 }
 }).join('');}
let body=blocks(marked.lexer(md));
for(let i=0;i<equations.length;i++){
 let {value,block}=equations[i];
 if(block&&value.includes('r=r_{term}'))value='\\begin{aligned}'+value.replace('r=r_{term}','r={}&r_{term}').replace('-0.08','\\\\ &-0.08').replace('-5\\frac','\\\\ &-5\\frac')+'\\end{aligned}';
 const tex=block?'\n\\begin{center}\\begin{adjustbox}{max width=\\linewidth}$\\displaystyle '+value.trim()+'$\\end{adjustbox}\\end{center}\n':'$'+value+'$';
 body=body.split(`ZZFORMULA${i}ZZ`).join(tex);
}
const fontdir=path.join(root,'.venv/lib/python3.12/site-packages/matplotlib/mpl-data/fonts/ttf/');
const preamble=String.raw`\documentclass[12pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[provide=*,spanish]{babel}
\usepackage{amsmath,amsfonts,amssymb}
\usepackage{geometry}
\usepackage{hyperref}
\usepackage{enumitem}
\usepackage{microtype}
\usepackage{mathpazo}
\usepackage{xcolor,graphicx,booktabs,longtable,array,colortbl,adjustbox,pdflscape}
\usepackage{needspace,fvextra,xurl}
\geometry{margin=2.5cm}
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue,citecolor=blue}
\hypersetup{pdftitle={Control de aterrizaje de un vehículo 2D},pdfauthor={Andy Fuentes; Diederich Solís; Diego Linares},pdfsubject={CC3104 - Aprendizaje por Refuerzo; informe final}}
\setlist[itemize]{itemsep=2pt,topsep=4pt}
\setlist[enumerate]{itemsep=4pt,topsep=4pt}
\definecolor{Accent}{rgb}{0,0,1}
\definecolor{Pale}{gray}{0.95}
\setcounter{secnumdepth}{2}
\setcounter{tocdepth}{2}
\setlength{\emergencystretch}{3em}
\sloppy
\DefineVerbatimEnvironment{CodeBlock}{Verbatim}{fontsize=\footnotesize,breaklines=true,breakanywhere=true}
\begin{document}
%-------------------- CARÁTULA --------------------
\hypersetup{pageanchor=false}
\begin{titlepage}
  \centering
  \vspace*{2cm}
  {\LARGE\textbf{Universidad del Valle de Guatemala}}\\[0.5cm]
  {\Large Facultad de Ingeniería}\\[1.5cm]
  {\Huge\textbf{Control de aterrizaje\\ de un vehículo 2D}}\\[0.5cm]
  {\Large\textbf{Informe final}}\\[0.4cm]
  {\Large\textbf{CC3104 -- Aprendizaje por Refuerzo}}\\[2cm]
  \textbf{Integrantes:}\\[0.3cm]
  Andy Fuentes --- 22944\\
  Diederich Solís --- 22952\\
  Diego Linares --- 221256\\[2cm]
  \textbf{Fecha: 4 de noviembre de 2026}
  \vfill
\end{titlepage}
\hypersetup{pageanchor=true}
\pagenumbering{arabic}
\tableofcontents
\clearpage
BODY
\end{document}
`;
fs.writeFileSync('tmp/pdfs/informe_final.tex',preamble.replaceAll('FONTDIR',fontdir).replace('BODY',()=>body));
fs.writeFileSync('tmp/pdfs/conversion.json',JSON.stringify({source:'docs/informe_final.md',equations:equations.length,sections:marked.lexer(md).filter(x=>x.type==='heading'&&x.depth===2).length,tables:marked.lexer(md).filter(x=>x.type==='table').length},null,2));
console.log('LaTeX created:',equations.length,'math expressions');
