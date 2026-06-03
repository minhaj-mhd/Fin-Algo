import re

filepath = r'C:\Users\loq\Desktop\Trading\finalgo\templates\vanguard_v2.html'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

pattern = re.compile(
    r'(const sId = (p|t|stat|trade)\.strategy_id;)\s*'
    r'const stratBadge = sId !== null && sId !== undefined && sId !== -1 \? \s*'
    r'`<a href="/strategy/\$\{sId\}" class="badge bg-dark text-info border border-info border-opacity-25 text-decoration-none" style="font-size: 8px; vertical-align: middle;">S\$\{sId\}</a>` : \s*'
    r'`<a href="/strategy/0" class="badge bg-dark text-dim border border-secondary border-opacity-10 text-decoration-none" style="font-size: 8px; vertical-align: middle;">AI</a>`;'
)

def replacer(match):
    prefix = match.group(1)
    obj_name = match.group(2)
    
    return f"""{prefix}
                        const isEns_{obj_name} = {obj_name}.is_ensemble === 1 || {obj_name}.is_ensemble === true;
                        let stratBadge = sId !== null && sId !== undefined && sId !== -1 ? 
                            `<a href="/strategy/${{sId}}" class="badge bg-dark text-info border border-info border-opacity-25 text-decoration-none" style="font-size: 8px; vertical-align: middle;">S${{sId}}</a>` : 
                            `<a href="/strategy/0" class="badge bg-dark text-dim border border-secondary border-opacity-10 text-decoration-none" style="font-size: 8px; vertical-align: middle;">AI</a>`;
                        
                        if (isEns_{obj_name} && sId !== null && sId !== undefined && sId !== -1) {{
                            stratBadge = `<a href="/strategy/0" class="badge bg-dark text-dim border border-secondary border-opacity-10 text-decoration-none" style="font-size: 8px; vertical-align: middle;">AI</a> ` + stratBadge;
                        }}"""

new_content = pattern.sub(replacer, content)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(new_content)
    
print("Update applied to vanguard_v2.html")
