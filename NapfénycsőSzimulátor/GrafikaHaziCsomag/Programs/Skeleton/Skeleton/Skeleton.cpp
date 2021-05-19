//=============================================================================================
// Mintaprogram: Z?ld h?romsz?g. Ervenyes 2019. osztol.
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
// Nev    : Kalmar Gabor Gyorgy
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

enum MaterialType {ROUGH, REFLECTIVE};

struct Material {
	vec3 ka, kd, ks;
	float shininess;
	MaterialType type;
	vec3 F0;
	Material(MaterialType t) { type = t; }
};

struct RoughMaterial : Material {
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) : Material(ROUGH) {
		ka = _kd * M_PI;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
	}
};

vec3 operator/(vec3 num, vec3 denom) {
	return vec3(num.x / denom.x, num.y / denom.y, num.z / denom.z);
}


struct ReflectiveMaterial : Material {
	ReflectiveMaterial(vec3 n, vec3 kappa) : Material(REFLECTIVE) {
		vec3 one(1, 1, 1);
		F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
	}
};

struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

mat4 T(mat4 m) {
	mat4 t;
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			t[i][j] = m[j][i];
		}
	}
	return t;
}

const mat4 hyperboloid = mat4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1);
const mat4 paraboloid = mat4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0);
const mat4 cylinder = mat4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1);
const mat4 ellipsoid = mat4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1);


class Quadratic : public Intersectable {
protected:
	mat4 Q;
	float top, bottom;
public:
	Quadratic(mat4 matrix, Material* _material, float _bottom, float _top) {
		Q = matrix;
		material = _material;
		bottom = _bottom;
		top = _top;
	}

	vec3 Gradf(vec4 r) {
		r.w = 1;
		vec4 g = r * Q * 2;
		return vec3(g.x, g.y, g.z);
	}

	void Scale(vec3 v) {
		Q = ScaleMatrix(v) * Q * T(ScaleMatrix(v));
	}

	void Transform(vec3 v) {
		Q = TranslateMatrix(v) * Q * T(TranslateMatrix(v));
	}

	void Rotate(float angle, vec3 v) {
		Q = RotationMatrix(angle, v)*Q*T(RotationMatrix(angle, v));
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec4 norm_dir = vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0);
		vec4 norm_start = vec4(ray.start.x, ray.start.y, ray.start.z, 1);
		float a = dot(norm_dir *Q, norm_dir);
		float b = dot(norm_start * Q, norm_dir) + dot(norm_dir *Q, norm_start);
		float c = dot(norm_start * Q, norm_start);
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / (2.0f * a);
		float t2 = (-b - sqrt_discr) / (2.0f * a);
		
		if (t1 <= 0) return hit;

		hit.t = (t2 > 0) ? t2 : t1;
		float t_temp = (t2 > 0) ? t1 : t2;

		hit.position = ray.start + ray.dir * hit.t;
		if ((hit.position.z > top) || hit.position.z < bottom) {
			hit.t = t_temp;
			hit.position = ray.start + ray.dir * hit.t;
			if ((hit.position.z > top) || hit.position.z < bottom)
				return Hit();
			hit.normal = (-1) * normalize(Gradf(vec4(hit.position.x, hit.position.y, hit.position.z, 1)));
			hit.material = material;
			return hit;
		}
		hit.normal = normalize(Gradf(vec4(hit.position.x, hit.position.y, hit.position.z, 1)));
		hit.material = material;
		
		return hit;
	}
};

class Camera {
	vec3 eye, lookat, right, up;
	float fov;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
		eye = _eye; lookat = _lookat; fov = _fov;
		vec3 w = eye - lookat;
		float windowSize = length(w) * tanf(fov / 2);
		right = normalize(cross(vup, w)) * windowSize;
		up = normalize(cross(w, right)) * windowSize;
	}

	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2 * (X + 0.5f) / windowWidth - 1) + up * (2 * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
};


struct Light {
	vec3 direction;
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction);
		Le = _Le;
	}
};

float rnd() { return (float)rand() / RAND_MAX; }
const float epsilon = 0.0001f;

class Scene {
	std::vector<Intersectable*> objects;
	std::vector<vec3> RandomPoints;
	Camera camera;
	vec3 La = vec3(0.4f, 0.4f, 0.4f);
	vec3 Sky = vec3(0.74f, 0.76f, 0.99f);
	Light *Sun = new Light(vec3(100, 100, 100),vec3(0.99f, 0.99f, 0.5f));

public:
	void build() {
		vec3 eye = vec3(0,9, 1), vup = vec3(0, 0, 1), lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180 * 1.3;
		camera.set(eye, lookat, vup, fov);
		RandPoint();
		
		vec3 kd1(0.3f, 0.42f, 0.0f), kd2(0.1f, 0.2f, 0.3f), kd3(0.3f,0.0f,0.0f), kd4(0.5,0.2,0), ks(2, 2, 2);
		vec3 n_gold(0.17, 0.35, 1.5), kappa_gold(3.1, 2.7, 1.9);
		vec3 n_silver(0.14, 0.16, 0.13), kappa_silver(4.1, 2.3, 3.1);

		Material* material_yellow = new RoughMaterial(kd1, ks, 80);
		Material* material_blue = new RoughMaterial(kd2, ks, 50);
		Material* material_red = new RoughMaterial(kd3, ks, 80);
		Material* material_orange = new RoughMaterial(kd4, ks, 80);
		Material* material_gold = new ReflectiveMaterial(n_gold, kappa_gold);
		Material* material_silver = new ReflectiveMaterial(n_silver, kappa_silver);

		Quadratic* Room = new Quadratic(ellipsoid, material_blue, -5, 4.5);
		Quadratic* SunlightTube = new Quadratic(hyperboloid, material_silver, 4.5, 8);
		Quadratic* Hyperboloid = new Quadratic(hyperboloid, material_yellow, -5, 1.5); //-4.5
		Quadratic* Paraboloid = new Quadratic(paraboloid, material_gold, -5, 10);
		Quadratic* Cylinder = new Quadratic(cylinder, material_red, -5, -3);

		Room->Scale(vec3(0.1, 0.1, 0.2));
		SunlightTube->Transform(vec3(0, 0, -2.5)); // x: (+)jobbra, (-)balra      y: (+)hatra, (-)elore    Z: (+)le, (-)fel
		SunlightTube->Scale(vec3(0.25, 0.25, 0.4));
		Hyperboloid->Transform(vec3(-4, 13, 1));
		Hyperboloid->Scale(vec3(5, 5, 1));
		Hyperboloid->Rotate(M_PI / 10, vec3(0, 1, 0));
		Paraboloid->Transform(vec3(0.5, 3, 0));
		Paraboloid->Scale(vec3(2, 1, 1));
		Cylinder->Transform(vec3(2, 0, 1));
		Cylinder->Rotate(-M_PI / 10, vec3(0, 1, 0));

		objects.push_back(Room);
		objects.push_back(SunlightTube);
		objects.push_back(Hyperboloid);
		objects.push_back(Paraboloid);
		objects.push_back(Cylinder);
	}

	void render(std::vector<vec4>& image) {
		long time = glutGet(GLUT_ELAPSED_TIME);

		for (int Y = 0; Y < windowHeight; Y++)
		{
			#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++)
			{
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects)
		{
			Hit hit = object->intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t)) bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {
		for (Intersectable* object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}


	void RandPoint() {
		float x = 100;
		float y = 100;
		while (RandomPoints.size()<=8) {
			x = 2 * sqrtf(rand() % (19 + 1)) - sqrtf(19);
			y = 2 * sqrtf(rand() % (19 + 1)) - sqrtf(19);
			if (sqrtf(pow(x, 2) + pow(y, 2)) <= 19)
				RandomPoints.push_back(vec3(x, y, 4.5f));
		}
	}


	vec3 trace(Ray ray, int depth = 0) {
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;

		float r = sqrtf(19);
		vec3 outRadiance = (0, 0, 0);

		if (hit.material->type == ROUGH) {
			outRadiance = hit.material->ka * La; 
			for (int i = 0; i < RandomPoints.size(); i++) {
				vec3 pointDir = RandomPoints[i] - hit.position;
				float omega = r * r * M_PI / RandomPoints.size() * (dot(vec3(0, 0, -1), normalize(-pointDir))) / dot(pointDir, pointDir);
				float cosTheta = dot(hit.normal, normalize(pointDir));
				Ray nextRay(hit.position + hit.normal * epsilon, pointDir);
				Hit nextHit = firstIntersect(nextRay);
				if (cosTheta > 0 && !shadowIntersect(nextRay)) {
					outRadiance = outRadiance + Sun->Le * hit.material->kd * cosTheta * omega;
					vec3 halfway = normalize(-ray.dir + pointDir);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0)
						outRadiance = outRadiance + Sky * Sun->Le * hit.material->ks * powf(cosDelta, hit.material->shininess) * omega;
				}
				else if (cosTheta > 0 && nextHit.material->type == REFLECTIVE) {
					vec3 sun = Reflective(nextHit, nextRay, outRadiance, depth + 1);
					if (cosTheta > 0) {
						vec3 halfway = normalize(-ray.dir + pointDir);
						float cosDelta = dot(hit.normal, halfway);
						outRadiance = outRadiance + sun * hit.material->kd * cosTheta * omega;
						if (cosDelta > 0) 
							outRadiance = outRadiance + sun * hit.material->ks * powf(cosDelta, hit.material->shininess) * omega;
					}
				}
			}
		}
		else {
			outRadiance = Reflective(hit, ray, outRadiance, depth + 1);
		}
		return outRadiance;
	}


	vec3 Reflective(Hit hit, Ray ray, vec3 outRadiance, int depth) {
		if (depth > 2) return La;
		if (hit.t < 0){
			return Sky + Sun->Le * pow(dot(ray.dir, Sun->direction), 5);
		}
		vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
		float cosa = -dot(ray.dir, hit.normal);
		vec3 one(1, 1, 1);
		vec3 F = hit.material->F0 + (one - hit.material->F0) * pow(1.0f - cosa, 5);
		outRadiance = outRadiance + trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth + 1) * F;
		return outRadiance;
	}
};

Scene scene;
GPUProgram gpuProgram;

const char* const vertexSource = R"(
	#version 330
	precision highp float;

	layout(location = 0) in vec2 cVertexPosition;
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1)) / 2;
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1);
	}
)";

// fragment shader in GLSL
const char* const fragmentSource = R"(
	#version 330
	precision highp float;
	
	uniform sampler2D textureUnit;
	in vec2 texcoord;
	out vec4 fragmentColor;

	void main() { fragmentColor = texture(textureUnit, texcoord); }
)";


class FullScreenTextQuad {
	unsigned int vao = 0, textureId = 0;
public:
	FullScreenTextQuad(int windowWidth, int windowHeight) {
		glGenVertexArrays(1, &vao);	// get 1 vao id
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;
		glGenBuffers(1, &vbo);	// Generate 1 buffer
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		float vertexCoords[] = { -1,-1,1,-1,1,1,-1,1 };
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

		glGenTextures(1, &textureId);
		glBindTexture(GL_TEXTURE_2D, textureId);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}

	void LoadTexture(std::vector<vec4>& image) {
		glBindTexture(GL_TEXTURE_2D, textureId);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, &image[0]);
	}


	void Draw() {
		glBindVertexArray(vao);
		int location = glGetUniformLocation(gpuProgram.getId(), "textureUnit");
		const unsigned int textureUnit = 0;
		if (location >= 0) {
			glUniform1i(location, textureUnit);
			glActiveTexture(GL_TEXTURE0 + textureUnit);
			glBindTexture(GL_TEXTURE_2D, textureId);
		}
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}

};

FullScreenTextQuad *fullScreenTextQuad;

void onInitialization() {

	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();
	fullScreenTextQuad = new FullScreenTextQuad(windowWidth, windowHeight);

	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	std::vector<vec4> image(windowHeight * windowWidth);
	scene.render(image);
	fullScreenTextQuad->LoadTexture(image);
	fullScreenTextQuad->Draw();
	glutSwapBuffers();
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}


int pickedControlPoint = -1;
// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
}


// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
}
