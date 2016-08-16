import os

def make_catalogue(catalogue_dir):

    # make catalogue.html

    files = [v for v in os.listdir(catalogue_dir) if v.endswith('.jpg')]
    files.sort()

    files_list = ','.join(files)

    html = """
<script>
var files_list = '""" + files_list + """';
var files = files_list.split(',');
</script>
<style>
.catalogue {
    width: 95%;
    height: auto;
}

.catalogue-entry {
    display: inline-block;
    list-style: none inside;
}

.catalogue-entry img {
    width: 220px;
}

.catalogue-entry figcaption {
    font-size: 0.5rem;
}

</style>
<script>
document.addEventListener('DOMContentLoaded', function(ev) {
  //var files = decodeURIComponent(location.hash.substr(1)).split(',');
  var catalogue = document.querySelector('#catalogue');
  for (var i = 0; i < files.length; i++) {
    var file = files[i];
    var shortname = file.split('/')[1];
    var li = document.createElement('li');
    li.className = 'catalogue-entry';
    li.innerHTML = '<figure><img src="' + file + '"></img><figcaption>' + shortname + '</figcaption></figure>';
    catalogue.appendChild(li);
  }
});
</script>
<table border="0">
    <tr>
        <td>
<ul id='catalogue' class='catalogue'>
</ul>
        </td>
    </tr>
    </table>
</body>
"""

    open(os.path.join(catalogue_dir, 'catalogue.html'),'wb').write(html)

if '__main__' == __name__:
    import argparse
    parser = argparse.ArgumentParser(description='catalog html maker')
    parser.add_argument('catalogue_dir', type=str)
    args = parser.parse_args()
    catalogue_dir=args.catalogue_dir
    make_catalogue(catalogue_dir)

# Emacs:
# Local Variables:
# mode: python
# c-basic-offset: 4
# End:
# vim: sw=4 sts=4 ts=8 et ft=python
