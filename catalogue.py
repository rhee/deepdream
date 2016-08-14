import os
output_dir = 'layers'

# make catalogue.html
files_list = ','.join([os.path.join(output_dir, v) for v in os.listdir(output_dir) if v.endswith('.jpg')])

from urllib import quote
url = 'catalogue.template.html#' + quote(files_list)
open('catalogue.html','w').write('''
<iframe style="width:100%;border:0;overflow:hidden;" src="''' + url + '''"
  onload="javascript:this.style.height=(this.contentWindow.document.body.scrollHeight+20)+'px'">
</iframe>
''')
