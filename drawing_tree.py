import StringIO
import pydot
from IPython.core.display import Image
from sklearn.tree import export_graphviz
def draw_tree(classifier, feature_names):
    dot_buf = StringIO.StringIO()
    export_graphviz(classifier,  
                    out_file=dot_buf, 
                    feature_names = feature_names) # classifier: clf_model, feature_names: df.columns
    graph = pydot.graph_from_dot_data(dot_buf.getvalue())[0]
    image = graph.create_png()
    return Image(image)