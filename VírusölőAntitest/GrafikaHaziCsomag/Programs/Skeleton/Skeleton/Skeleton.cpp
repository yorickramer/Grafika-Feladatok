//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Kalmár Gábor
// Neptun : XCCGBS
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

template<class T> struct Dnum {
	float f;
	T d;
	Dnum(float f0 = 0, T d0 = T()) { f = f0; d = d0;}
	Dnum operator+(Dnum r) { return Dnum(f + r.f, d + r.d); }
	Dnum operator-(Dnum r) { return Dnum(f - r.f, d - r.d); }
	Dnum operator*(Dnum r) { return Dnum(f * r.f, f * r.d + d * r.f); }
	Dnum operator/(Dnum r) { return Dnum(f / r.f, (r.f * d - r.d * f) / r.f / r.f); }
};

template<class T> Dnum<T> Exp(Dnum<T> g) { return Dnum<T>(expf(g.f), expf(g.f) * g.d); }
template<class T> Dnum<T> Sin(Dnum<T> g) { return Dnum<T>(sinf(g.f), cosf(g.f) * g.d); }
template<class T> Dnum<T> Cos(Dnum<T> g) { return Dnum<T>(cosf(g.f), -sinf(g.f) * g.d); }
template<class T> Dnum<T> Tan(Dnum<T> g) { return Sin(g) / Cos(g); }
template<class T> Dnum<T> Sinh(Dnum<T> g) { return Dnum<T>(sinh(g.f), cosh(g.f) * g.d); }
template<class T> Dnum<T> Cosh(Dnum<T> g) { return Dnum<T>(cosh(g.f), sinh(g.f) * g.d); }
template<class T> Dnum<T> Tanh(Dnum<T> g) { return Sinh(g) / Cosh(g); }
template<class T> Dnum<T> Log(Dnum<T> g) { return Dnum<T>(logf(g.f), g.d / g.f); }
template<class T> Dnum<T> Pow(Dnum<T> g, float n) { return Dnum<T>(powf(g.f, n), n * powf(g.f, n - 1) * g.d); }

typedef Dnum<vec2> Dnum2;
const int tessellationLevel = 20;

struct Camera {
	vec3 wEye, wLookat, wVup;
	float fov, asp, fp, bp;
public:
	Camera() {
		asp = (float)windowWidth / windowHeight;
		fov = 75.0f * (float)M_PI / 180.0f;
		fp = 1; bp = 30;
	}
	mat4 V() {
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(wEye * (-1)) * mat4(	u.x, v.x, w.x, 0,
													u.y, v.y, w.y, 0,
													u.z, v.z, w.z, 0,
													0,0, 0, 1		);
	}
	mat4 P() {
		return mat4(1 / (tan(fov / 2) * asp), 0, 0, 0,
					0, 1 / tan(fov / 2), 0, 0,
					0, 0, -(fp + bp) / (bp - fp), -1,
					0, 0, -2 * fp * bp / (bp - fp), 0);
	}
};

struct Material {
	vec3 kd, ks, ka;
	float shininess;
};

struct Light {
	vec3 La, Le;
	vec4 wLightPos;
	void Animate(float t) {	}
};

struct Tetra {
	vec3 p1, p2, p3, p4;
	float m;
public:
	Tetra(vec3 p1_, vec3 p2_, vec3 p3_, float m_) { 
		p1 = p1_; 
		p2 = p2_; 
		p3 = p3_; 
		m = m_; 
		p4 = vec3((p1.x + p2.x + p3.x) / 3, (p1.y + p2.y + p3.y) / 3, (p1.z + p2.z + p3.z) / 3) + normalize(cross((p1 - p2), (p1 - p3))) * m;
	}
	Tetra(){}
};

//done
class ZebraTexture : public Texture {
public:
	ZebraTexture(const int width, const int height, const vec4 color1, const vec4 color2){
		std::vector<vec4> image(width * height);
		for (int x = 0; x < width; x++)
			for (int y = 0; y < height; y++){
				if (x % 2 == 0)
					image[y * width + x] = color1;
				else
					image[y * width + x] = color2;
			}
		create(width, height, image, GL_NEAREST);
	}
};


struct RenderState {
	mat4 MVP, M, Minv, V, P; 
	Material* material;
	std::vector<Light> lights;
	Texture* texture;
	vec3 wEye;
};


class Shader : public GPUProgram {
public:
	virtual void Bind(RenderState state) = 0;

	void setUniformMaterial(const Material& material, const std::string& name){
		setUniform(material.kd, name + ".kd");
		setUniform(material.ks, name + ".ks");
		setUniform(material.ka, name + ".ka");
		setUniform(material.shininess, name + ".shininess");
	}

	void setUniformLight(const Light& light, const std::string& name) {
		setUniform(light.La, name + ".La");
		setUniform(light.Le, name + ".Le");
		setUniform(light.wLightPos, name + ".wLightPos");
	}
};


class GouraudShader : public Shader {
	const char* vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};
		
		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform mat4  MVP, M, Minv;  // MVP, Model, Model-inverse
		uniform Light[8] lights;     // light source direction 
		uniform int   nLights;		 // number of light sources
		uniform vec3  wEye;          // pos of eye
		uniform Material  material;  // diffuse, specular, ambient ref

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space

		out vec3 radiance;		    // reflected radiance, sugársûrûség

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			// radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;	//vilagkordinátarendszerve, ekkor kell az illuminációt elvégezni
			vec3 V = normalize(wEye * wPos.w - wPos.xyz); //kamera es a pont felé mutató vektor
			vec3 N = normalize((Minv * vec4(vtxNorm, 0)).xyz);
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein

			radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				radiance += material.ka * lights[i].La + (material.kd * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
		}
	)";

	// fragment shader in GLSL
	const char* fragmentSource = R"(
		#version 330
		precision highp float;

		in  vec3 radiance;      // interpolated radiance
		out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			fragmentColor = vec4(radiance, 1);
		}
	)";


public:
	GouraudShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		Use(); 		// make this program run
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniformMaterial(*state.material, "material");

		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};


class PhongShader : public Shader {
	const char* vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform vec3  wEye;         // pos of eye

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal;		    // normal in world space
		out vec3 wView;             // view in world space
		out vec3 wLight[8];		    // light dir in world space
		out vec2 texcoord;

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			// vectors for radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		    texcoord = vtxUV;
		}
	)";

	// fragment shader in GLSL
	const char* fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material;
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform sampler2D diffuseTexture;

		in  vec3 wNormal;       // interpolated world sp normal
		in  vec3 wView;         // interpolated world sp view
		in  vec3 wLight[8];     // interpolated world sp illum dir
		in  vec2 texcoord;
		
        out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein
			vec3 texColor = texture(diffuseTexture, texcoord).rgb;
			vec3 ka = material.ka * texColor;
			vec3 kd = material.kd * texColor;

			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				// kd and ka are modulated by the texture
				radiance += ka * lights[i].La + 
                           (kd * texColor * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	PhongShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		Use(); 		// make this program run
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniform(*state.texture, std::string("diffuseTexture"));
		setUniformMaterial(*state.material, "material");

		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};

//---------------------------
class NPRShader : public Shader {
	//---------------------------
	const char* vertexSource = R"(
		#version 330
		precision highp float;

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform	vec4  wLightPos;
		uniform vec3  wEye;         // pos of eye

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal, wView, wLight;				// in world space
		out vec2 texcoord;

		void main() {
		   gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
		   vec4 wPos = vec4(vtxPos, 1) * M;
		   wLight = wLightPos.xyz * wPos.w - wPos.xyz * wLightPos.w;
		   wView  = wEye * wPos.w - wPos.xyz;
		   wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		   texcoord = vtxUV;
		}
	)";

	// fragment shader in GLSL
	const char* fragmentSource = R"(
		#version 330
		precision highp float;

		uniform sampler2D diffuseTexture;

		in  vec3 wNormal, wView, wLight;	// interpolated
		in  vec2 texcoord;
		out vec4 fragmentColor;    			// output goes to frame buffer

		void main() {
		   vec3 N = normalize(wNormal), V = normalize(wView), L = normalize(wLight);
		   float y = (dot(N, L) > 0.5) ? 1 : 0.5;
		   if (abs(dot(N, V)) < 0.2) fragmentColor = vec4(0, 0, 0, 1);
		   else						 fragmentColor = vec4(y * texture(diffuseTexture, texcoord).rgb, 1);
		}
	)";
public:
	NPRShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		Use(); 		// make this program run
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniform(*state.texture, std::string("diffuseTexture"));
		setUniform(state.lights[0].wLightPos, "wLightPos");
	}
};


struct VertexData { 
	vec3 position, normal;
	vec2 texcoord;
};


class Geometry {
protected:
	unsigned int vao, vbo;
public:
	Geometry() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	}
	virtual void Draw() = 0;
	~Geometry() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};


class ParamSurface : public Geometry {
	unsigned int nVtxPerStrip, nStrips;
public:
	ParamSurface() { nVtxPerStrip = nStrips = 0; }
	virtual void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) = 0;

	VertexData GenVertexData(float u, float v) {
		VertexData vtxData;
		vtxData.texcoord = vec2(u, v);
		Dnum2 X, Y, Z;
		Dnum2 U(u, vec2(1, 0)), V(v, vec2(0, 1));
		eval(U, V, X, Y, Z);

		vtxData.position = vec3(X.f, Y.f, Z.f);
		vec3 drdU(X.d.x, Y.d.x, Z.d.x), drdV(X.d.y, Y.d.y, Z.d.y); //.f tartalmazza függvényértéket, .d a deriváltat
		vtxData.normal = cross(drdU, drdV);
		return vtxData;
	}

	void create(int N = tessellationLevel, int M = tessellationLevel) { //Megvan ,N,M, hany szor hanyra daraboljuk fel
		nVtxPerStrip = (M + 1) * 2;
		nStrips = N;
		std::vector<VertexData> vtxData;	// vertices on the CPU
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
				vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
			}
		}
		glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
		glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
		glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
		// attribute array, components/attribute, component type, normalize?, stride, offset
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}

	void Draw() {
		glBindVertexArray(vao);
		for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i * nVtxPerStrip, nVtxPerStrip);
	}
};


class Tractricoid : public ParamSurface {
public:
	Tractricoid() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		const float height = 3.0f;
		U = U * height;
		V = V * 2 * M_PI;
		X = Cos(V) / Cosh(U);
		Y = Sin(V) / Cosh(U);
		Z = Tanh(U) - U;
	}
};

//sinus fv
class Virus : public ParamSurface {
	float time;
public:
	Virus(float t) {
		time = t; 
		create();
	}
	void SetTime(float t) { time = t; }

	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {

		Dnum2 T(time, vec2(0, 0));
		Dnum2 K = Sin((V * 5) * T); //sinf((30 * v + 4 * u)*time); //sinf(30 * v + 4 * u) * cosf(time) * 0.1 + 1;Sphere
		Dnum2 O(1, vec2(0, 0));
		Dnum2 R = K * 0.1 + O;
		U = U * 2.0f * (float)M_PI;
		V = V * (float)M_PI;

		X = Cos(U) * Sin(V) * R;
		Y = Sin(U) * Sin(V) * R;
		Z = Cos(V) * R;
	}
};

class Sphere : public ParamSurface {

public:
	Sphere() {create();}

	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {

		U = U * 2.0f * (float)M_PI;
		V = V * (float)M_PI;

		X = Cos(U) * Sin(V);
		Y = Sin(U) * Sin(V);
		Z = Cos(V);
	}
};



class Tetraeder : public Geometry {
	vec3 p1, p2, p3, p4;
	vec3 normal123, normal234, normal341, normal412;
	float m;
public:
	Tetraeder(vec3 p1_, vec3 p2_, vec3 p3_, float m_) {
		p1 = p1_;
		p2 = p2_;
		p3 = p3_;
		m = m_;
		create();
	}

	void setM(float m_) { m = m_; }

	vec3 Getp1() { return p1; }
	vec3 Getp2() { return p2; }
	vec3 Getp3() { return p3; }

	void create() {
		
		normal123 = normalize(cross((p1 - p2), (p1 - p3)));
		p4 = vec3((p1.x + p2.x + p3.x) / 3, (p1.y + p2.y + p3.y) / 3, (p1.z + p2.z + p3.z) / 3) + normal123 * m;
		normal234 = normalize(cross((p2 - p3), (p2 - p4)));
		normal341 = normalize(cross((p3 - p4), (p3 - p1)));
		normal412 = normalize(cross((p4 - p1), (p4 - p2)));

		std::vector<VertexData> vtxData;
		
		VertexData vtx1;
		VertexData vtx2;
		VertexData vtx3;
		VertexData vtx4;

		vtx1.normal = normal123; vtx1.position = p1; vtx1.texcoord = vec2(0.01, 0.25);
		vtx2.normal = normal234; vtx2.position = p2; vtx2.texcoord = vec2(0.25, 0.5);
		vtx3.normal = normal341; vtx3.position = p3; vtx3.texcoord = vec2(0.51, 0.75);
		vtx4.normal = normal412; vtx4.position = p4; vtx4.texcoord = vec2(0.75, 0.99);

		vtxData.push_back(vtx1);
		vtxData.push_back(vtx2);
		vtxData.push_back(vtx3);

		vtxData.push_back(vtx2);
		vtxData.push_back(vtx3);
		vtxData.push_back(VertexData(vtx4));

		vtxData.push_back(VertexData(vtx3));
		vtxData.push_back(VertexData(vtx4));
		vtxData.push_back(VertexData(vtx1));

		vtxData.push_back(VertexData(vtx4));
		vtxData.push_back(VertexData(vtx1));
		vtxData.push_back(VertexData(vtx2));


		glBufferData(GL_ARRAY_BUFFER, sizeof(VertexData) * 4 * 3, &vtxData[0], GL_STATIC_DRAW);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
		glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
		glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0

		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}

	void Draw() {
		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 12);
	}
};


struct Object {
	Shader* shader;
	Material* material;
	Texture* texture;
	Geometry* geometry;
	vec3 scale, translation, rotationAxis;
	float rotationAngle;
public:
	Object(Shader* _shader, Material* _material, Texture* _texture, Geometry* _geometry) :
		scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
		shader = _shader;
		texture = _texture;
		material = _material;
		geometry = _geometry;
	}

	virtual void SetModelingTransform(mat4& M, mat4& Minv) {
		M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
	}

	void Draw(RenderState state) {
		mat4 M, Minv;
		SetModelingTransform(M, Minv);
		state.M = M;
		state.Minv = Minv;
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		state.texture = texture;
		shader->Bind(state);
		geometry->Draw();
	}

	void SetRotationAngle(float f) { rotationAngle = f; }

	virtual void Animate(float tstart, float tend) {
		rotationAngle = 0.6f * tend;
	}
};

void TetraObj(std::vector<Object*>* objects, Tetra* t , Material* material, Shader* shader, Texture* texture, vec3 pos) {

	Geometry* tetra = new Tetraeder(t->p1, t->p2, t->p3, t->m);
	Object* tetraObject1 = new Object(shader, material, texture, tetra);
	tetraObject1->translation = pos;
	tetraObject1->rotationAxis = vec3(0, 1, 1);
	tetraObject1->scale = vec3(1.0f, 1.0f, 1.0f);
	objects->push_back(tetraObject1);
}


void TetraRec(std::vector<Tetra*>* tetras, float m , int depth) {

	if (depth > 2)
		return;

	std::vector<Tetra*> temp_tetras;
	for (Tetra* tet : *tetras) {
		vec3 p1_ = (tet->p1 + tet->p2) / 2;
		vec3 p2_ = (tet->p2 + tet->p4) / 2;
		vec3 p3_ = (tet->p4 + tet->p1) / 2;
		temp_tetras.push_back(new Tetra(p1_, p2_, p3_, m));
		p1_ = (tet->p1 + tet->p3) / 2;
		p2_ = (tet->p3 + tet->p4) / 2;
		p3_ = (tet->p4 + tet->p1) / 2;
		temp_tetras.push_back(new Tetra(p1_, p2_, p3_, -m));
		p1_ = (tet->p2 + tet->p3) / 2;
		p2_ = (tet->p3 + tet->p4) / 2;
		p3_ = (tet->p4 + tet->p2) / 2;
		temp_tetras.push_back(new Tetra(p1_, p2_, p3_, m));
		p1_ = (tet->p1 + tet->p3) / 2;
		p2_ = (tet->p3 + tet->p2) / 2;
		p3_ = (tet->p2 + tet->p1) / 2;
		temp_tetras.push_back(new Tetra(p1_, p2_, p3_, m));
	}
	for (Tetra* tet : temp_tetras)
		tetras->push_back(tet);
			

	TetraRec(tetras, m/2, depth + 1);
}


void AntitestMaker(std::vector<Object*>* objects, vec3 p1, vec3 p2, vec3 p3, float m, Material* material, Shader* shader, Texture* texture, vec3 pos) {
	std::vector<Tetra*> tetras;
	Tetra* tat = new Tetra(p1,p2,p3,1);
	tetras.push_back(tat);
	
	TetraRec(&tetras, m, 1);

	for (Tetra* tet : tetras)
		TetraObj(objects, tet, material, shader, texture, pos);
}




std::vector<vec3> NormalV(float u, float v, float time) {
	std::vector<vec3> res;
	vec3 position;
	vec3 normal;

	Dnum2 U(u, vec2(1, 0)), V(v, vec2(0, 1));

	Dnum2 T(time, vec2(0, 0));
	Dnum2 K = Sin((V * 5) * T);
	Dnum2 O(1, vec2(0, 0));
	Dnum2 R = K * 0.1 + O;

	U = U * 2.0f * (float)M_PI;
	V = V * (float)M_PI;
	Dnum2 X = Cos(U) * Sin(V) * R;
	Dnum2 Y = Sin(U) * Sin(V) * R;
	Dnum2 Z = Cos(V) * R;

	position = vec3(X.f, Y.f, Z.f) * 2;
	vec3 drdU(X.d.x, Y.d.x, Z.d.x), drdV(X.d.y, Y.d.y, Z.d.y); //.f tartalmazza függvényértéket, .d a deriváltat
	normal = normalize(cross(drdU, drdV));
	res.push_back(position);
	res.push_back(normal);
	return res;
}


void VirusMaker(std::vector<Object*>* objects, Material* material, Shader* shader, Texture* texture, float t, vec3 pos) {
	
	Geometry* virus = new Virus(t);
	
	Object* sphereObject1 = new Object(shader, material, texture, virus);
	sphereObject1->translation = pos; //jobb, felul, elol-hatul

	sphereObject1->rotationAxis = vec3(0, 1, 1);
	sphereObject1->scale = vec3(2.0f, 2.0f, 2.0f);
	objects->push_back(sphereObject1);
}


class Scene {
	std::vector<Object*> objects;
	std::vector<Light> lights;
	Camera camera;
	Material* material1;
	Shader* gouraudShader;
	Shader* nprShader;
	Shader* phongShader;

	//virus
	Texture* virus_texture;
	vec3 posVirus = vec3(0, 0, 0);
	float timevirus = 7000;

	//antitest
	Texture* antitest_texture;
	vec3 pos_antitest = vec3(0, 0, -5);
	vec3 speed_antitest;
	int dir_antitest = 0;
	float timeantitest = 100;
	float height = 1;
	float height_diff = 0.02f;
	bool down;

	//background
	Object* backgroundObject;
	
public:
	void Build() {

		phongShader = new PhongShader();
		gouraudShader = new GouraudShader();
		nprShader = new NPRShader();

		Material* backgroundmaterial = new Material;
		backgroundmaterial->kd = vec3(0.4f, 0.4f, 0.6f);
		backgroundmaterial->ks = vec3(4, 4, 4);
		backgroundmaterial->ka = vec3(0.1f, 0.1f, 0.1f);
		backgroundmaterial->shininess = 80;

		material1 = new Material;
		material1->kd = vec3(0.8f, 0.6f, 0.4f);
		material1->ks = vec3(0.3f, 0.3f, 0.3f);
		material1->ka = vec3(0.2f, 0.2f, 0.2f);
		material1->shininess = 30;

		virus_texture = new ZebraTexture(20, 20, vec4(0.203, 0.152, 0.152, 1), vec4(0.843, 0.670, 0.337));
		antitest_texture = new ZebraTexture(20, 20, vec4(0.4, 0, 0.5, 1), vec4(0.4, 0, 0.5, 1));

		Texture* texturebackground = new ZebraTexture(10, 10, vec4(0.27f, 0.55f, 0.655f, 1), vec4(0.27f, 0.55f, 0.655f, 1));
		Geometry* background = new Sphere();

		backgroundObject = new Object(gouraudShader, backgroundmaterial, texturebackground, background);
		backgroundObject->scale = vec3(20, 20, 20);

		camera.wEye = vec3(0, 0, 8);
		camera.wLookat = vec3(0, 0, 0);
		camera.wVup = vec3(0, 1, 0);

		lights.resize(3);
		lights[0].wLightPos = vec4(30, 30, 30, 1);
		lights[0].La = vec3(0.1f, 0.1f, 1);
		lights[0].Le = vec3(3, 0, 0);

		lights[1].wLightPos = vec4(0, 0, 30, 1);
		lights[1].La = vec3(0.2f, 0.2f, 0.2f);
		lights[1].Le = vec3(0, 3, 0);

		lights[2].wLightPos = vec4(-15, -30, 15, 1);
		lights[2].La = vec3(0.1f, 0.1f, 0.1f);
		lights[2].Le = vec3(0, 0, 3);
	}

	void Render() {
		RenderState state;
		state.wEye = camera.wEye;
		state.V = camera.V();
		state.P = camera.P();
		state.lights = lights;
		for (Object* obj : objects) obj->Draw(state);
		backgroundObject->Draw(state);
	}

	void KeyBoard(int d) { dir_antitest = d; }

	void AntitestCalculate(float tend) {

		if (height >= 2)
			down = true;
		if (height <= 1)
			down = false;
		if (down)
			height = height - height_diff;
		else
			height = height + height_diff;

		timeantitest = timeantitest + tend;
		if (timeantitest > 100) {
			timeantitest = 0;
			if (dir_antitest == 0)
				speed_antitest = vec3((float)(rand() % 11 - 5), (float)(rand() % 11 - 5), (float)(rand() % 11 - 5));
			else if (dir_antitest == 1) //x pozitív fele megy
				speed_antitest = vec3((float)(rand() % 11 - 2), (float)(rand() % 11 - 5), (float)(rand() % 11 - 5));
			else if (dir_antitest == 2) //y pozitív fele megy
				speed_antitest = vec3((float)(rand() % 11 - 5), (float)(rand() % 11 - 2), (float)(rand() % 11 - 5));
			else if (dir_antitest == 3) //z pozitív fele megy
				speed_antitest = vec3((float)(rand() % 11 - 5), (float)(rand() % 11 - 5), (float)(rand() % 11 - 2));
			else if (dir_antitest == 4) //X negativ fele megy
				speed_antitest = vec3((float)(rand() % 11 - 8), (float)(rand() % 11 - 5), (float)(rand() % 11 - 5));
			else if (dir_antitest == 5) //Y negativ fele megy
				speed_antitest = vec3((float)(rand() % 11 - 5), (float)(rand() % 11 - 8), (float)(rand() % 11 - 5));
			else if (dir_antitest == 6) //Z negativ fele megy
				speed_antitest = vec3((float)(rand() % 11 - 5), (float)(rand() % 11 - 5), (float)(rand() % 11 - 8));
			speed_antitest = speed_antitest / 500;
		}
		pos_antitest = pos_antitest + speed_antitest;
	}


	vec4 qmul(vec4 q1, vec4 q2) {
		vec3 d1(q1.x, q1.y, q1.z);
		vec3 d2(q2.x, q2.y, q2.z);
		vec3 tmp = d2 * q1.w + d1 * q2.w + cross(d1, d2);
		return vec4(tmp.x,tmp.y, tmp.z, q1.w * q2.w - dot(d1, d2));
	}

	vec4 quaternion(float ang, vec3 axis) {
		vec3 d = normalize(axis);
		d.x = d.x * sinf(ang / 2);
		d.y = d.y * sinf(ang / 3);
		d.z = d.z * sinf(ang / 5);
		
		return vec4(d.x, d.y, d.z, cosf(ang));
	}

	vec3 Rotate(vec3 u, vec4 q) {
		vec4 qinv(-q.x, -q.y, -q.z, q.w);
		vec4 qr = qmul(qmul(q, vec4(u.x, u.y, u.z, 0)), qinv);
		return vec3(qr.x, qr.y, qr.z);
	}


	void VirusCalculate(float tend) {
		timevirus = timevirus + tend;
		float sec = timevirus / 1000;
		vec4 q = quaternion(sec, vec3(1,1,1));
		posVirus = Rotate(vec3(0,0,-5), q);
	}


	void Animate(float tstart, float tend) {
		for (unsigned int i = 0; i < lights.size(); i++) { lights[i].Animate(tend); }
		for (Object* obj : objects)
			delete obj;
		objects.clear();

		AntitestCalculate(tend);
		VirusCalculate(tend);
		AntitestMaker(&objects, vec3(0, 0, 0), vec3(0, 1, 1), vec3(0, 1, 0), height/2, material1, nprShader, antitest_texture, pos_antitest);
		VirusMaker(&objects, material1, phongShader, virus_texture, tend, posVirus);
		for (Object* obj : objects) obj->Animate(tstart, tend);
	}
};

Scene scene;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	scene.Build();
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	scene.Render();
	glutSwapBuffers();
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	switch (key){
	case 'x': scene.KeyBoard(1); break;
	case 'y': scene.KeyBoard(2); break;
	case 'z': scene.KeyBoard(3); break;
	case 'X': scene.KeyBoard(4); break;
	case 'Y': scene.KeyBoard(5); break;
	case 'Z': scene.KeyBoard(6); break;
	default:scene.KeyBoard(0); break;
	}
}


void onKeyboardUp(unsigned char key, int pX, int pY) { }

void onMouse(int button, int state, int pX, int pY) { }

void onMouseMotion(int pX, int pY) {
}

void onIdle() {
	static float tend = 0;
	const float dt = 0.1f;
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

	for (float t = tstart; t < tend; t += dt) {
		float Dt = fmin(dt, tend - t);
		scene.Animate(t, t + Dt);
	}
	glutPostRedisplay();
}