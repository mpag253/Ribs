

$study = 'Human_Aging';
$subject = 'AGING001';
$condition = 'EIsupine';
 
#$study = 'Human_Lung_Atlas';
#$subject ='P2BRP139-H6229'; #  'P2BRP043-H245';  #
#$condition = 'EIsupine'; 
 
$lung_dir = '/hpc/mpag253/Torso/surface_fitting/'
$ribs_dir = '/hpc/mpag253/Ribs/point_clouds/output/'
$ribspath = $ribs_dir.$study.'/'.$subject.'/'.$condition.'/Ribs/Data';
$lungpath = $lung_dir.$study.'/'.$subject.'/'.$condition.'/Lung/SurfaceFEMesh';

#gfx read node $fitpath.'/Torso_fitted' reg mesh_TorsoFitted;
#gfx read elem $fitpath.'/Torso_fitted' reg mesh_TorsoFitted;
gfx read node $lungpath.'/'.'Left_fitted' reg mesh_LungLeft;
gfx read elem $lungpath.'/'.'Left_fitted' reg mesh_LungLeft;
gfx read node $lungpath.'/'.'Right_fitted' reg mesh_LungRight;
gfx read elem $lungpath.'/'.'Right_fitted' reg mesh_LungRight;

#gfx read data $datapath.'/ribs_data_'.$subject.'_'.$condition;
gfx read data $ribspath.'/ribs_data_T01L';
gfx read data $ribspath.'/ribs_data_T02L';
gfx read data $ribspath.'/ribs_data_T03L';
gfx read data $ribspath.'/ribs_data_T04L';
gfx read data $ribspath.'/ribs_data_T05L';
gfx read data $ribspath.'/ribs_data_T06L';
gfx read data $ribspath.'/ribs_data_T07L';
gfx read data $ribspath.'/ribs_data_T08L';
gfx read data $ribspath.'/ribs_data_T09L';
gfx read data $ribspath.'/ribs_data_T10L';
gfx read data $ribspath.'/ribs_data_T11L';
gfx read data $ribspath.'/ribs_data_T12L';
gfx read data $ribspath.'/ribs_data_T01R';
gfx read data $ribspath.'/ribs_data_T02R';
gfx read data $ribspath.'/ribs_data_T03R';
gfx read data $ribspath.'/ribs_data_T04R';
gfx read data $ribspath.'/ribs_data_T05R';
gfx read data $ribspath.'/ribs_data_T06R';
gfx read data $ribspath.'/ribs_data_T07R';
gfx read data $ribspath.'/ribs_data_T08R';
gfx read data $ribspath.'/ribs_data_T09R';
gfx read data $ribspath.'/ribs_data_T10R';
gfx read data $ribspath.'/ribs_data_T11R';
gfx read data $ribspath.'/ribs_data_T12R';

#gfx cre mat lung_surface ambient 0.4 0.4 0.4 diffuse 0.7 0.7 0.7 specular 0.5 0.5 0.5 alpha 0.4;
gfx mod g_e mesh_LungLeft surface mat muscle;
gfx mod g_e mesh_LungRight surface mat muscle;
gfx mod g_e mesh_LungLeft line mat black;
gfx mod g_e mesh_LungRight line mat black;
#gfx mod g_e mesh_TorsoFitted surface mat tissue;
#gfx mod g_e mesh_TorsoFitted node_points glyph sphere general size "4*4*4" material blue;
#gfx mod g_e surface_Torso data_points glyph point size "4*4*4" material green;

#gfx mod g_e Rib_ALL data_points glyph point material white;
gfx mod g_e Rib_T01L data_points glyph point material white;
gfx mod g_e Rib_T02L data_points glyph point material white;
gfx mod g_e Rib_T03L data_points glyph point material white;
gfx mod g_e Rib_T04L data_points glyph point material white;
gfx mod g_e Rib_T05L data_points glyph point material white;
gfx mod g_e Rib_T06L data_points glyph point material white;
gfx mod g_e Rib_T07L data_points glyph point material white;
gfx mod g_e Rib_T08L data_points glyph point material white;
gfx mod g_e Rib_T09L data_points glyph point material white;
gfx mod g_e Rib_T10L data_points glyph point material white;
gfx mod g_e Rib_T11L data_points glyph point material white;
gfx mod g_e Rib_T12L data_points glyph point material white;
gfx mod g_e Rib_T01R data_points glyph point material white;
gfx mod g_e Rib_T02R data_points glyph point material white;
gfx mod g_e Rib_T03R data_points glyph point material white;
gfx mod g_e Rib_T04R data_points glyph point material white;
gfx mod g_e Rib_T05R data_points glyph point material white;
gfx mod g_e Rib_T06R data_points glyph point material white;
gfx mod g_e Rib_T07R data_points glyph point material white;
gfx mod g_e Rib_T08R data_points glyph point material white;
gfx mod g_e Rib_T09R data_points glyph point material white;
gfx mod g_e Rib_T10R data_points glyph point material white;
gfx mod g_e Rib_T11R data_points glyph point material white;
gfx mod g_e Rib_T12R data_points glyph point material white;

gfx edit scene;
gfx cre wind;
#gfx node_tool edit select;

#open com;edit_derivatives
